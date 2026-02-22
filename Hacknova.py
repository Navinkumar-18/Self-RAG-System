"""
Autonomous Self-Improving RAG System
- Vector retrieval → LLM answer → Confidence evaluation
- If low confidence: Wikipedia fetch → add to FAISS → regenerate
- When info is not in dataset: fallback to LLM general knowledge.
- Fully offline except Wikipedia API.
"""

import re
import streamlit as st
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
CONFIDENCE_THRESHOLD = 0.6
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3

# ==============================
# EMBEDDING MODEL (offline)
# ==============================

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# INITIAL KNOWLEDGE BASE
# ==============================

INITIAL_DOCUMENTS = [
    "A vector database stores high-dimensional embeddings for similarity search.",
    "LangChain is a framework for building applications using large language models.",
    "Retrieval Augmented Generation retrieves relevant documents before generating answers.",
    "FAISS is a library developed by Facebook AI for efficient similarity search.",
    "Embeddings convert text into numerical vectors for semantic comparison.",
]
INITIAL_DOC_IDS = [f"doc_{i}" for i in range(len(INITIAL_DOCUMENTS))]

# ==============================
# SESSION STATE
# ==============================

def init_session_state():
    if "expanded_documents" not in st.session_state:
        st.session_state.expanded_documents = []
    if "expanded_doc_ids" not in st.session_state:
        st.session_state.expanded_doc_ids = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_expanded_chunk_ids" not in st.session_state:
        st.session_state.last_expanded_chunk_ids = set()

init_session_state()

# ==============================
# FAISS INDEX (rebuilt each run with initial + expanded)
# ==============================

def get_documents_and_ids():
    """Merge initial and expanded documents."""
    docs = INITIAL_DOCUMENTS + st.session_state.expanded_documents
    ids = INITIAL_DOC_IDS + st.session_state.expanded_doc_ids
    return docs, ids

def build_index(documents, embedding_model):
    """Build FAISS index from document list."""
    embeddings = embedding_model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

# ==============================
# RETRIEVAL
# ==============================

def retrieve(query, index, doc_ids, documents, embedding_model, top_k=TOP_K):
    """Retrieve top-k documents by similarity."""
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)
    results = []
    for idx in I[0]:
        if idx < len(doc_ids):
            results.append((doc_ids[idx], documents[idx]))
    return results

# ==============================
# OLLAMA HELPERS
# ==============================

def call_ollama(prompt, stream=False):
    """Single call to Ollama generate API."""
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": stream},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def generate_answer(query, retrieved_docs):
    """Generate answer strictly from retrieved context."""
    context_text = "\n\n".join(
        [f"[{doc_id}] {content}" for doc_id, content in retrieved_docs]
    )
    prompt = f"""You are an AI assistant in a RAG system.

STRICT RULES:
1. Answer ONLY using the retrieved context below.
2. Do NOT use any outside knowledge.
3. If the context does not contain enough information, say exactly:
   "The retrieved documents do not contain sufficient information to answer this question."

User question: {query}

Retrieved context:
{context_text}

Answer (based only on the context above):"""
    return call_ollama(prompt)

# Exact phrase we use when context is insufficient (for detection)
INSUFFICIENT_MSG = "The retrieved documents do not contain sufficient information to answer this question."

def answer_with_llama_direct(prompt_text):
    """Use the Llama model (Ollama) directly to answer the given prompt. No RAG context."""
    return call_ollama(prompt_text)

def evaluate_confidence(query, answer, retrieved_docs):
    """Ask the model: is the answer fully supported by context? Return score 0-1."""
    context_text = "\n\n".join(
        [f"[{doc_id}] {content}" for doc_id, content in retrieved_docs]
    )
    prompt = f"""Question: {query}

Retrieved context:
{context_text}

Generated answer: {answer}

Is the answer fully supported by the retrieved context? Respond with only a confidence score between 0 and 1 (e.g. 0.85 or 0.3)."""
    response = call_ollama(prompt)
    # Parse first number in [0,1]
    match = re.search(r"0?\.\d+|1\.0?", response)
    if match:
        return float(match.group())
    return 0.5  # fallback

# ==============================
# WIKIPEDIA EXPANSION
# ==============================

def fetch_wikipedia_summary(query):
    """Fetch summary text from Wikipedia. Returns None if not found."""
    title = query.replace(" ", "_")[:100]
    url = f"{WIKI_SUMMARY_URL}/{title}"
    headers = {"User-Agent": "AutonomousRAG/1.0 (Educational; Streamlit)"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("extract") or None
    except Exception:
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def expand_knowledge(query, embedding_model):
    """
    Fetch Wikipedia summary, chunk, embed, add to FAISS and session state.
    Returns list of (doc_id, content) for the new chunks.
    """
    summary = fetch_wikipedia_summary(query)
    if not summary:
        return []

    chunks = chunk_text(summary)
    if not chunks:
        return []

    base_id = f"wiki_{len(st.session_state.expanded_doc_ids)}"
    new_ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
    st.session_state.expanded_doc_ids.extend(new_ids)
    st.session_state.expanded_documents.extend(chunks)
    return list(zip(new_ids, chunks))

# ==============================
# MAIN FLOW
# ==============================

def run_rag_flow(query):
    """
    Execute: retrieve → answer → confidence → [if low: expand → retrieve → answer].
    Returns dict with answer, confidence, expansion_triggered, retrieved_docs,
    expanded_chunk_ids (set), final_retrieved_docs (after expansion if any).
    """
    model = get_embedding_model()
    documents, doc_ids = get_documents_and_ids()
    index = build_index(documents, model)

    # First retrieval and answer
    retrieved = retrieve(query, index, doc_ids, documents, model)
    if not retrieved:
        fallback = answer_with_llama_direct(query)
        return {
            "answer": fallback,
            "confidence": 0.0,
            "expansion_triggered": False,
            "retrieved_docs": [],
            "expanded_chunk_ids": set(),
            "final_retrieved_docs": [],
            "answer_source": "general_knowledge",
            "fallback_answer": fallback,
        }

    answer = generate_answer(query, retrieved)
    confidence = evaluate_confidence(query, answer, retrieved)

    expanded_chunk_ids = set()
    final_retrieved = retrieved

    if confidence < CONFIDENCE_THRESHOLD:
        new_chunks = expand_knowledge(query, model)
        for doc_id, _ in new_chunks:
            expanded_chunk_ids.add(doc_id)
        st.session_state.last_expanded_chunk_ids = expanded_chunk_ids

        # Rebuild index and retrieve again
        documents, doc_ids = get_documents_and_ids()
        index = build_index(documents, model)
        final_retrieved = retrieve(query, index, doc_ids, documents, model)
        answer = generate_answer(query, final_retrieved)

    # If still no info in dataset, answer from LLM general knowledge
    answer_source = "context"
    fallback_answer = None
    if answer.strip().startswith(INSUFFICIENT_MSG.split(".")[0]) or answer.strip() == INSUFFICIENT_MSG:
        fallback_answer = answer_with_llama_direct(query)
        answer = fallback_answer
        answer_source = "general_knowledge"

    return {
        "answer": answer,
        "confidence": confidence,
        "expansion_triggered": confidence < CONFIDENCE_THRESHOLD,
        "retrieved_docs": retrieved,
        "expanded_chunk_ids": expanded_chunk_ids,
        "final_retrieved_docs": final_retrieved,
        "answer_source": answer_source,
        "fallback_answer": fallback_answer,
    }

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Autonomous Self-Improving RAG", layout="wide")
st.title("🧠 Autonomous Self-Improving RAG System")

st.caption(
    "Context-based answers when possible. If the dataset has no relevant info, the LLM answers from general knowledge."
)

query = st.text_input("Ask a question:", placeholder="e.g. Vector database, RAG, FAISS...")

if st.button("Generate Answer") and query:
    with st.spinner("Retrieving, generating, and evaluating..."):
        result = run_rag_flow(query)

    st.session_state.last_answer = result["answer"]
    st.session_state.last_result = result

    # Confidence and expansion status
    col1, col2 = st.columns(2)
    with col1:
        conf = result["confidence"]
        color = "green" if conf >= CONFIDENCE_THRESHOLD else "orange"
        st.markdown(f"**Confidence score:** :{color}[**{conf:.2f}**]")
    with col2:
        if result["expansion_triggered"]:
            st.warning("⚠️ Knowledge expansion was triggered (confidence < 0.6)")
        else:
            st.success("✓ No expansion needed")

    st.subheader("Answer")
    if result.get("answer_source") == "general_knowledge":
        st.info("📡 **Source: Llama model (direct)** — no sufficient information in the dataset; answer generated directly from the model for your prompt.")
    st.write(result["answer"])

    expanded_ids = result["expanded_chunk_ids"]
    final_docs = result.get("final_retrieved_docs", [])
    if final_docs:
        st.subheader("Retrieved chunks (used for answer)")
    for doc_id, content in final_docs:
        if doc_id in expanded_ids:
            st.markdown(f"**{doc_id}** *(newly added from Wikipedia)*")
            st.markdown(f"> {content}")
        else:
            st.markdown(f"**{doc_id}**")
            st.write(content)
        st.write("---")

    if result["expansion_triggered"] and expanded_ids:
        with st.expander("Newly added knowledge (Wikipedia)"):
            docs, ids = get_documents_and_ids()
            for i, doc_id in enumerate(ids):
                if doc_id in expanded_ids:
                    st.markdown(f"**{doc_id}**")
                    st.write(docs[i])
                    st.write("---")

# Optional: show current knowledge base size
with st.sidebar:
    st.subheader("Knowledge base")
    docs, ids = get_documents_and_ids()
    st.write(f"Total chunks: **{len(docs)}**")
    st.write(f"Initial: {len(INITIAL_DOCUMENTS)}, Expanded: {len(st.session_state.expanded_documents)}")
