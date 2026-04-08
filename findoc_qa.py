"""
FinDoc QA — 金融文档智能问答系统
Step 1: PDF 解析 + 智能分块 + Embedding + 向量存储
"""

import os
import json
import fitz  # pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import time

# ============================================================
# 1. PDF 解析：提取文本 + 表格检测
# ============================================================
def parse_pdf(pdf_path: str) -> list[dict]:
    """解析 PDF，按页提取文本，标记是否含表格"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        tables = page.find_tables()
        has_table = len(tables.tables) > 0 if tables else False

        pages.append({
            "page_num": i + 1,
            "text": text.strip(),
            "has_table": has_table,
            "char_count": len(text.strip())
        })
    doc.close()
    return pages


# ============================================================
# 2. 智能分块
# ============================================================
def chunk_pages(pages: list[dict],
                chunk_size: int = 800,
                chunk_overlap: int = 200) -> list[dict]:
    """将页面文本分块，保留元数据"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        if not page["text"]:
            continue
        splits = splitter.split_text(page["text"])
        for j, text in enumerate(splits):
            chunks.append({
                "id": f"p{page['page_num']}_c{j}",
                "text": text,
                "page_num": page["page_num"],
                "has_table": page["has_table"],
                "chunk_index": j
            })
    return chunks


# ============================================================
# 3. Embedding + 存入 ChromaDB
# ============================================================
def build_vector_store(chunks: list[dict],
                       collection_name: str = "findoc",
                       persist_dir: str = "data/chroma_db"):
    """用 BGE-small 做 embedding，存入 ChromaDB"""
    print("🔄 Loading embedding model (BGE-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"page_num": c["page_num"],
                  "has_table": c["has_table"],
                  "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"🔄 Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"✅ Embedding done in {time.time() - start:.1f}s")

    # 存入 ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    print(f"✅ Stored {len(texts)} chunks in ChromaDB → {persist_dir}")
    return collection, model


# ============================================================
# 4. 检索测试
# ============================================================
def search(query: str, collection, model, top_k: int = 5):
    """检索最相关的 chunks"""
    query_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )
    return results


# ============================================================
# 运行完整 pipeline
# ============================================================
PDF_PATH = "data/pdfs/apple_10k_2024.pdf"

# Step 1: 解析 PDF
print("=" * 60)
print("📄 Step 1: Parsing PDF...")
pages = parse_pdf(PDF_PATH)
print(f"   Pages: {len(pages)}")
print(f"   Total chars: {sum(p['char_count'] for p in pages):,}")
print(f"   Pages with tables: {sum(p['has_table'] for p in pages)}")

# Step 2: 分块
print("\n📦 Step 2: Chunking...")
chunks = chunk_pages(pages)
print(f"   Chunks: {len(chunks)}")
avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
print(f"   Avg chunk length: {avg_len:.0f} chars")

# 保存 chunks 到 JSON
os.makedirs("data", exist_ok=True)
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print("   Saved → data/chunks.json")

# Step 3: Embedding + 向量存储
print("\n🧠 Step 3: Building vector store...")
collection, model = build_vector_store(chunks)

# Step 4: 检索测试
print("\n🔍 Step 4: Search test...")
test_queries = [
    "What was Apple's total revenue in 2024?",
    "What are the main risk factors?",
    "How much did Apple spend on research and development?",
]
for q in test_queries:
    print(f"\n   Q: {q}")
    results = search(q, collection, model)
    for i, (doc, score) in enumerate(
        zip(results["documents"][0], results["distances"][0])
    ):
        print(f"   [{i+1}] (dist={score:.3f}) {doc[:120]}...")

print("\n✅ Pipeline complete!")

# ============================================================
# Step 5: RAG Pipeline — 检索 + LLM 生成带引用的回答
# ============================================================
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

def rag_answer(question: str, collection, embed_model, top_k: int = 5):
    """检索相关文档 + 调用 GPT 生成带引用的回答"""
    # 1. 检索
    results = search(question, collection, embed_model, top_k=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 2. 构建 context
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        context_parts.append(f"[Source {i+1}, Page {meta['page_num']}] (relevance: {1-dist:.2f})\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Prompt
    system_prompt = """You are a financial document analyst. Answer the user's question based ONLY on the provided context from the document. 
Rules:
- Cite specific sources using [Source N] format
- If the context doesn't contain enough information, say so clearly
- For numerical data, quote exact figures from the document
- Be concise but thorough"""

    user_prompt = f"""Context from Apple 10-K filing:

{context}

---

Question: {question}

Answer with citations:"""

    # 4. 调用 GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=800
    )
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return {
        "question": question,
        "answer": answer,
        "sources": [{"page": m["page_num"], "distance": d, "text": doc[:200]}
                    for m, d, doc in zip(metas, distances, docs)],
        "tokens_used": tokens_used
    }

# ============================================================
# Step 5: 测试 RAG 问答
# ============================================================
print("\n" + "=" * 60)
print("🤖 Step 5: RAG Q&A Test...")

test_questions = [
    "What was Apple's total net sales in fiscal year 2024?",
    "What are the top 3 risk factors Apple faces?",
    "How much did Apple spend on R&D in 2024 compared to 2023?",
    "What was Apple's net income in 2024?",
    "How does Apple describe its competition in the market?",
]

all_results = []
for q in test_questions:
    print(f"\n{'─' * 50}")
    print(f"Q: {q}")
    result = rag_answer(q, collection, model)
    print(f"\nA: {result['answer']}")
    print(f"\n   📊 Tokens used: {result['tokens_used']}")
    print(f"   📄 Top source: Page {result['sources'][0]['page']} (relevance: {1-result['sources'][0]['distance']:.2f})")
    all_results.append(result)

# 保存结果
with open("eval/rag_test_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results saved → eval/rag_test_results.json")
print(f"💰 Total tokens: {sum(r['tokens_used'] for r in all_results):,}")

# ============================================================
# Step 6: 评估系统 — Faithfulness + Relevance + 消融实验
# ============================================================
import re

# ---------- 6a. Ground Truth 测试集 ----------
GROUND_TRUTH = [
    {
        "question": "What was Apple's total net sales in fiscal year 2024?",
        "gold_answer": "$391,035 million",
        "keywords": ["391,035", "391"],
        "category": "numerical"
    },
    {
        "question": "How much did Apple spend on R&D in 2024?",
        "gold_answer": "$31,370 million",
        "keywords": ["31,370", "31.4"],
        "category": "numerical"
    },
    {
        "question": "What was Apple's net income in 2024?",
        "gold_answer": "$93,736 million",
        "keywords": ["93,736", "93.7"],
        "category": "numerical"
    },
    {
        "question": "What is Apple's total number of employees?",
        "gold_answer": "Approximately 164,000",
        "keywords": ["164,000", "164"],
        "category": "numerical"
    },
    {
        "question": "Who is Apple's CEO?",
        "gold_answer": "Tim Cook",
        "keywords": ["Tim Cook", "Cook"],
        "category": "factual"
    },
    {
        "question": "What was Apple's gross margin in 2024?",
        "gold_answer": "46.2%",
        "keywords": ["46.2", "46%", "180,683"],
        "category": "numerical"
    },
    {
        "question": "How much cash did Apple return to shareholders in 2024?",
        "gold_answer": "Over $100 billion through dividends and share repurchases",
        "keywords": ["dividend", "repurchas", "buyback", "100 billion", "94,949", "15,234"],
        "category": "numerical"
    },
    {
        "question": "What are Apple's main product categories?",
        "gold_answer": "iPhone, Mac, iPad, Wearables/Home/Accessories, Services",
        "keywords": ["iPhone", "Mac", "iPad", "Wearables", "Services"],
        "category": "factual"
    },
    {
        "question": "In which countries does Apple have retail stores?",
        "gold_answer": "Multiple countries including the U.S., with stores in 26 countries",
        "keywords": ["retail", "store", "country", "countries"],
        "category": "factual"
    },
    {
        "question": "What was iPhone revenue in 2024?",
        "gold_answer": "$201,183 million",
        "keywords": ["201,183", "201"],
        "category": "numerical"
    },
    {
        "question": "What was Services revenue in 2024?",
        "gold_answer": "$96,169 million",
        "keywords": ["96,169", "96"],
        "category": "numerical"
    },
    {
        "question": "What credit rating does Apple have?",
        "gold_answer": "AA+ from S&P / Aa1 from Moody's",
        "keywords": ["AA", "Aa1", "credit", "rating"],
        "category": "factual"
    },
]

# ---------- 6b. 评估函数 ----------
def eval_keyword_hit(answer: str, keywords: list[str]) -> bool:
    """答案中是否包含至少一个关键词"""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def eval_faithfulness(answer: str, sources: list[dict]) -> float:
    """简易 faithfulness: 答案中的数字是否出现在 source 文本中"""
    numbers_in_answer = set(re.findall(r'[\d,]+\.?\d*', answer))
    if not numbers_in_answer:
        return 1.0  # 没有数字 = 无法验证 = 给满分
    source_text = " ".join(s["text"] for s in sources).lower()
    source_numbers = set(re.findall(r'[\d,]+\.?\d*', source_text))
    matched = numbers_in_answer & source_numbers
    return len(matched) / len(numbers_in_answer) if numbers_in_answer else 1.0

def eval_relevance(distances: list[float]) -> float:
    """top-k 检索的平均相关度 (1 - cosine distance)"""
    return sum(1 - d for d in distances) / len(distances)

# ---------- 6c. 运行评估 ----------
print("\n" + "=" * 60)
print("📊 Step 6: Systematic Evaluation...")
print(f"   Test set: {len(GROUND_TRUTH)} questions\n")

eval_results = []
keyword_hits = 0
faith_scores = []
relevance_scores = []

for item in GROUND_TRUTH:
    q = item["question"]
    result = rag_answer(q, collection, model)

    hit = eval_keyword_hit(result["answer"], item["keywords"])
    faith = eval_faithfulness(result["answer"], result["sources"])
    rel = eval_relevance([s["distance"] for s in result["sources"]])

    keyword_hits += int(hit)
    faith_scores.append(faith)
    relevance_scores.append(rel)

    status = "✅" if hit else "❌"
    print(f"   {status} [{item['category']}] {q}")
    if not hit:
        print(f"      Expected keywords: {item['keywords']}")
        print(f"      Got: {result['answer'][:150]}...")

    eval_results.append({
        "question": q,
        "category": item["category"],
        "gold_answer": item["gold_answer"],
        "predicted_answer": result["answer"],
        "keyword_hit": hit,
        "faithfulness": faith,
        "relevance": rel,
        "tokens_used": result["tokens_used"],
        "sources": result["sources"]
    })

# ---------- 6d. 汇总指标 ----------
accuracy = keyword_hits / len(GROUND_TRUTH)
avg_faith = sum(faith_scores) / len(faith_scores)
avg_rel = sum(relevance_scores) / len(relevance_scores)

num_questions = [r for r in eval_results if r["category"] == "numerical"]
fact_questions = [r for r in eval_results if r["category"] == "factual"]
num_acc = sum(r["keyword_hit"] for r in num_questions) / len(num_questions)
fact_acc = sum(r["keyword_hit"] for r in fact_questions) / len(fact_questions)

print(f"\n{'─' * 50}")
print(f"📈 EVALUATION SUMMARY")
print(f"{'─' * 50}")
print(f"   Overall Accuracy:    {keyword_hits}/{len(GROUND_TRUTH)} = {accuracy:.1%}")
print(f"   Numerical Accuracy:  {num_acc:.1%} ({len(num_questions)} questions)")
print(f"   Factual Accuracy:    {fact_acc:.1%} ({len(fact_questions)} questions)")
print(f"   Avg Faithfulness:    {avg_faith:.3f}")
print(f"   Avg Relevance:       {avg_rel:.3f}")
print(f"   Total Tokens Used:   {sum(r['tokens_used'] for r in eval_results):,}")

# ---------- 6e. 保存评估结果 ----------
summary = {
    "overall_accuracy": accuracy,
    "numerical_accuracy": num_acc,
    "factual_accuracy": fact_acc,
    "avg_faithfulness": avg_faith,
    "avg_relevance": avg_rel,
    "total_questions": len(GROUND_TRUTH),
    "total_tokens": sum(r["tokens_used"] for r in eval_results),
    "details": eval_results
}
with open("eval/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n✅ Evaluation saved → eval/evaluation_results.json")

# ============================================================
# Step 7: 消融实验 — chunk_size × top_k
# ============================================================
print("\n" + "=" * 60)
print("🔬 Step 7: Ablation Study (chunk_size × top_k)...")

ablation_configs = [
    {"chunk_size": 500, "chunk_overlap": 100, "top_k": 3},
    {"chunk_size": 500, "chunk_overlap": 100, "top_k": 5},
    {"chunk_size": 500, "chunk_overlap": 100, "top_k": 8},
    {"chunk_size": 800, "chunk_overlap": 200, "top_k": 3},
    {"chunk_size": 800, "chunk_overlap": 200, "top_k": 5},  # baseline
    {"chunk_size": 800, "chunk_overlap": 200, "top_k": 8},
    {"chunk_size": 1200, "chunk_overlap": 300, "top_k": 3},
    {"chunk_size": 1200, "chunk_overlap": 300, "top_k": 5},
    {"chunk_size": 1200, "chunk_overlap": 300, "top_k": 8},
]

ablation_results = []

for config in ablation_configs:
    cs = config["chunk_size"]
    co = config["chunk_overlap"]
    tk = config["top_k"]
    label = f"chunk={cs}, overlap={co}, top_k={tk}"
    print(f"\n   ⚙️  {label}")

    # 重新分块
    test_chunks = chunk_pages(pages, chunk_size=cs, chunk_overlap=co)

    # 重新建向量库
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in test_chunks]
    ids = [c["id"] for c in test_chunks]
    metas = [{"page_num": c["page_num"],
              "has_table": c["has_table"],
              "chunk_index": c["chunk_index"]} for c in test_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)

    ab_client = chromadb.PersistentClient(path=f"data/chroma_ablation_{cs}_{co}")
    try:
        ab_client.delete_collection("ablation")
    except:
        pass
    ab_collection = ab_client.create_collection(
        name="ablation",
        metadata={"hnsw:space": "cosine"}
    )
    ab_collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metas
    )

    # 评估
    hits = 0
    faith_list = []
    rel_list = []
    tokens = 0

    for item in GROUND_TRUTH:
        result = rag_answer(item["question"], ab_collection, embed_model, top_k=tk)
        hit = eval_keyword_hit(result["answer"], item["keywords"])
        faith = eval_faithfulness(result["answer"], result["sources"])
        rel = eval_relevance([s["distance"] for s in result["sources"]])
        hits += int(hit)
        faith_list.append(faith)
        rel_list.append(rel)
        tokens += result["tokens_used"]

    acc = hits / len(GROUND_TRUTH)
    avg_f = sum(faith_list) / len(faith_list)
    avg_r = sum(rel_list) / len(rel_list)

    print(f"      Chunks: {len(test_chunks)} | Acc: {acc:.1%} | Faith: {avg_f:.3f} | Rel: {avg_r:.3f} | Tokens: {tokens:,}")

    ablation_results.append({
        "chunk_size": cs,
        "chunk_overlap": co,
        "top_k": tk,
        "num_chunks": len(test_chunks),
        "accuracy": acc,
        "faithfulness": avg_f,
        "relevance": avg_r,
        "tokens_used": tokens
    })

# ---------- 汇总表 ----------
print(f"\n{'─' * 70}")
print(f"{'chunk_size':>10} {'overlap':>8} {'top_k':>6} {'chunks':>7} {'acc':>8} {'faith':>8} {'rel':>8} {'tokens':>8}")
print(f"{'─' * 70}")
for r in ablation_results:
    marker = " ← baseline" if r["chunk_size"] == 800 and r["top_k"] == 5 else ""
    print(f"{r['chunk_size']:>10} {r['chunk_overlap']:>8} {r['top_k']:>6} {r['num_chunks']:>7} {r['accuracy']:>7.1%} {r['faithfulness']:>8.3f} {r['relevance']:>8.3f} {r['tokens_used']:>8,}{marker}")

# 保存
with open("eval/ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(ablation_results, f, indent=2)
print(f"\n✅ Ablation results saved → eval/ablation_results.json")