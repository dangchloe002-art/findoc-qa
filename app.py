"""
FinDoc QA — Streamlit Demo
"""
import os
import streamlit as st
import json
import math
import fitz
import chromadb
from collections import Counter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import opensearch_store as oss
from metadata import EXHIBIT_DOCS, TENK_ITEMS

# ============================================================
# 配置
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PERSIST_DIR = "data/chroma_db"
CHUNKS_PATH = "data/chunks.json"
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))

# ============================================================
# 缓存加载资源（只加载一次）
# ============================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection("findoc")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return collection, chunks

@st.cache_resource
def load_opensearch():
    client = oss.get_client(OPENSEARCH_HOST, OPENSEARCH_PORT)
    indices = sorted(client.indices.get(index=f"{oss.DEFAULT_INDEX}-*").keys())
    return client, indices

# ============================================================
# BM25
# ============================================================
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = documents
        self.doc_len = [len(d.split()) for d in documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.N = len(documents)
        self.df = Counter()
        self.doc_freqs = []
        for doc in documents:
            tokens = set(doc.lower().split())
            self.df.update(tokens)
            self.doc_freqs.append(Counter(doc.lower().split()))

    def score(self, query):
        query_tokens = query.lower().split()
        scores = []
        for i in range(self.N):
            s = 0
            for qt in query_tokens:
                if qt not in self.df:
                    continue
                idf = math.log((self.N - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
                tf = self.doc_freqs[i].get(qt, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += idf * (tf * (self.k1 + 1)) / denom
            scores.append(s)
        return scores

@st.cache_resource
def build_bm25(_chunks):
    texts = [c["text"] for c in _chunks]
    return BM25(texts)

# ============================================================
# 检索函数
# ============================================================
def dense_search(query, collection, embed_model, top_k=5):
    query_emb = embed_model.encode([query]).tolist()
    return collection.query(query_embeddings=query_emb, n_results=top_k)

def hybrid_search(query, collection, embed_model, bm25, chunks,
                  top_k=5, dense_weight=0.6):
    query_emb = embed_model.encode([query]).tolist()
    dense_results = collection.query(query_embeddings=query_emb, n_results=min(top_k * 3, len(chunks)))

    dense_scores = {}
    for doc_id, dist in zip(dense_results["ids"][0], dense_results["distances"][0]):
        dense_scores[doc_id] = 1 - dist

    bm25_raw = bm25.score(query)
    max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {chunks[i]["id"]: bm25_raw[i] / max_bm25 for i in range(len(chunks))}

    all_ids = set(dense_scores.keys()) | set(k for k, v in bm25_scores.items() if v > 0)
    fused = {}
    for doc_id in all_ids:
        d = dense_scores.get(doc_id, 0)
        b = bm25_scores.get(doc_id, 0)
        fused[doc_id] = dense_weight * d + (1 - dense_weight) * b

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    chunk_map = {c["id"]: c for c in chunks}

    return {
        "ids": [[r[0] for r in ranked]],
        "documents": [[chunk_map[r[0]]["text"] for r in ranked]],
        "distances": [[1 - r[1] for r in ranked]],
        "metadatas": [[{"page_num": chunk_map[r[0]]["page_num"],
                        "has_table": chunk_map[r[0]].get("has_table", False),
                        "chunk_index": chunk_map[r[0]]["chunk_index"]} for r in ranked]]
    }

# ============================================================
# RAG 生成
# ============================================================
def rag_answer(question, results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        context_parts.append(f"[Source {i+1}, Page {meta['page_num']}] (relevance: {1-dist:.2f})\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial document analyst. Answer based ONLY on the provided context. Cite sources using [Source N]. For numerical data, quote exact figures. Be concise but thorough."},
            {"role": "user", "content": f"Context from Apple 10-K filing:\n\n{context}\n\n---\nQuestion: {question}\nAnswer with citations:"}
        ],
        temperature=0, max_tokens=800
    )
    return response.choices[0].message.content, metas, distances, response.usage.total_tokens

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="FinDoc QA", page_icon="📄", layout="wide")

st.title("📄 FinDoc QA")
st.caption("Financial Document Question Answering — Apple 10-K (FY2024)")

# Load shared resources
embed_model = load_embed_model()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    backend = st.radio("Retrieval Backend", ["OpenSearch", "ChromaDB (original)"])
    top_k = st.slider("Top-K Results", 3, 10, 5)

    if backend == "OpenSearch":
        try:
            os_client, os_indices = load_opensearch()
        except Exception as e:
            st.error(f"Cannot reach OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}. "
                     f"Run `docker compose up -d` and `python build_opensearch_index.py`.\n\n{e}")
            st.stop()
        if not os_indices:
            st.error("No findoc-* index found. Run `python build_opensearch_index.py` first.")
            st.stop()
        os_index = st.selectbox("Index", os_indices, index=len(os_indices) - 1)
        search_mode = st.radio("Search Mode", ["Hybrid (BM25 + Dense)", "Dense Only", "BM25 Only"])
        dense_weight = 0.6
        if search_mode == "Hybrid (BM25 + Dense)":
            dense_weight = st.slider("Dense Weight", 0.3, 0.9, 0.6, 0.1)
        st.subheader("Metadata Filters")
        exclude_exhibits = st.checkbox("Exclude attached exhibit documents", value=True)
        section_names = [TENK_ITEMS[k] for k in TENK_ITEMS]
        sections = st.multiselect("Only these 10-K sections", section_names)
        tables_only = st.checkbox("Only pages with tables", value=False)
    else:
        collection, chunks = load_chroma()
        bm25 = build_bm25(chunks)
        search_mode = st.radio("Search Mode", ["Hybrid (BM25 + Dense)", "Dense Only"])
        if search_mode == "Hybrid (BM25 + Dense)":
            dense_weight = st.slider("Dense Weight", 0.3, 0.9, 0.6, 0.1)

    st.divider()
    st.header("📊 System Info")
    if backend == "OpenSearch":
        st.write(f"**Chunks:** {os_client.count(index=os_index)['count']}")
    else:
        st.write(f"**Chunks:** {collection.count()}")
    st.write(f"**Backend:** {backend}")
    st.write(f"**Embedding:** BGE-small-en-v1.5")
    st.write(f"**LLM:** GPT-4o-mini")

    st.divider()
    st.header("💡 Sample Questions")
    sample_questions = [
        "What was Apple's total revenue in 2024?",
        "What are the main risk factors?",
        "How much was spent on R&D?",
        "What was iPhone revenue?",
        "What is Apple's credit rating?",
    ]
    for sq in sample_questions:
        if st.button(sq, key=sq):
            st.session_state.question = sq

# 主界面
question = st.text_input("Ask a question about Apple's 10-K filing:",
                         value=st.session_state.get("question", ""),
                         placeholder="e.g. What was Apple's total revenue in 2024?")

if question:
    with st.spinner("🔍 Searching and generating answer..."):
        # Retrieval
        if backend == "OpenSearch":
            mode = {"Hybrid (BM25 + Dense)": "hybrid", "Dense Only": "dense",
                    "BM25 Only": "bm25"}[search_mode]
            filters = oss.metadata_filter(
                sections=sections or None,
                exclude_sections=[EXHIBIT_DOCS] if exclude_exhibits else None,
                has_table=True if tables_only else None)
            results = oss.search(os_client, question, embed_model, mode=mode, top_k=top_k,
                                 dense_weight=dense_weight, index=os_index, filters=filters)
        elif search_mode == "Hybrid (BM25 + Dense)":
            results = hybrid_search(question, collection, embed_model, bm25, chunks,
                                    top_k=top_k, dense_weight=dense_weight)
        else:
            results = dense_search(question, collection, embed_model, top_k=top_k)

        if not results["ids"][0]:
            st.warning("No chunks matched. Try relaxing the metadata filters.")
            st.stop()

        # 生成
        answer, metas, distances, tokens = rag_answer(question, results)

    # 答案
    st.markdown("### 💬 Answer")
    st.markdown(answer)

    # 指标
    col1, col2, col3 = st.columns(3)
    col1.metric("Sources Used", len(metas))
    col2.metric("Avg Relevance", f"{sum(1-d for d in distances)/len(distances):.1%}")
    col3.metric("Tokens Used", f"{tokens:,}")

    # 来源详情
    with st.expander("📄 Source Documents", expanded=False):
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], metas, distances)
        ):
            section = f" · {meta['section']}" if meta.get("section") else ""
            st.markdown(f"**[Source {i+1}] Page {meta['page_num']}{section}** (relevance: {1-dist:.2f})")
            st.text(doc[:500])
            st.divider()