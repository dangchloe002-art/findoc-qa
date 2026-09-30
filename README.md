# 📄 FinDoc QA — Financial Document Question Answering

An end-to-end RAG (Retrieval-Augmented Generation) system for financial document QA. Upload a 10-K filing and ask questions — the system retrieves relevant passages using hybrid search (BM25 + dense embeddings) and generates citation-grounded answers.

Retrieval runs on two interchangeable backends:
- **OpenSearch** (current): one index holds a BM25 text field, an HNSW k-NN vector field, and enriched metadata. Hybrid search is OpenSearch's native `hybrid` query with a min-max normalization search pipeline.
- **ChromaDB + custom BM25** (original): dense search in ChromaDB, BM25 in Python, weighted fusion on the client side.

## Architecture
```
PDF → PyMuPDF Parser → Recursive Chunking → Metadata Enrichment → BGE-small Embedding → OpenSearch index
                                           (10-K section, page,                          ├─ text      (BM25, English analyzer)
                                            table flag, company,                         ├─ embedding (k-NN, HNSW, cosine)
                                            fiscal year)                                 └─ metadata  (keyword / int / bool)
                                                                                                   ↓
User Question → hybrid query [BM25 + k-NN] → normalization pipeline (min-max, weighted mean)
             → metadata filters (section, tables, exhibits) → Top-K → GPT-4o-mini → Answer with [Source N] Citations
```

### Why OpenSearch
- **One engine for both signals.** The original BM25 scored all 721 chunks in Python on every query. OpenSearch serves BM25 from an inverted index and dense search from an HNSW graph, in one request.
- **Fusion inside the engine.** The `normalization-processor` min-max normalizes both score lists and combines them with weights `[1 - w, w]`. This is the same fusion rule as the original code, but tunable per request through a named search pipeline.
- **Metadata-aware retrieval.** Every chunk carries its 10-K section (inferred from the `Item N.` headings), page, table flag, company, and fiscal year. Filters apply to both the BM25 and the k-NN sub-query.
- **Finding from the metadata:** about half of all chunks (363 of 721 at chunk size 800) come from the exhibit documents attached after the signature page (indentures, stock plans). Filtering them out is one switch in the app and one experiment in the evaluation.

## Key Results

| Configuration | Accuracy | Faithfulness | Relevance |
|---|---|---|---|
| Dense only (baseline, chunk=800, top_k=5) | 83.3% | 0.622 | 0.735 |
| Dense only (chunk=1200, top_k=5) | 100.0% | 0.452 | 0.728 |
| Hybrid search (dense_weight=0.6) | 91.7% | — | — |

### Ablation: chunk_size × top_k (Dense Only)

| chunk_size | top_k | chunks | accuracy |
|---|---|---|---|
| 500 | 3 | 1008 | 58.3% |
| 500 | 5 | 1008 | 66.7% |
| 500 | 8 | 1008 | 75.0% |
| 800 | 3 | 721 | 83.3% |
| 800 | 5 | 721 | 83.3% |
| 800 | 8 | 721 | 91.7% |
| 1200 | 3 | 477 | 83.3% |
| 1200 | 5 | 477 | 100.0% |
| 1200 | 8 | 477 | 100.0% |

### Ablation: Dense Weight in Hybrid Search

| dense_weight | accuracy |
|---|---|
| 0.3–0.5 | 75.0% |
| 0.6–0.9 | 91.7% |

**Key Finding:** Larger chunks (1200 chars) preserve more context for financial tables, dramatically improving numerical QA accuracy. Hybrid search (BM25 + dense) fixes specific retrieval failures (e.g., net income) but requires weight tuning — pure BM25 dominance (weight < 0.5) degrades performance.

### OpenSearch results

Run `python run_opensearch_eval.py --build --llm` to fill this table. The script writes all numbers to `eval/opensearch_eval.json`.

| Index | Filter | Mode | Dense weight | Context hit rate | Accuracy (LLM) | Faithfulness | Latency (ms) |
|---|---|---|---|---|---|---|---|
| findoc-c1200 | none | hybrid | 0.6 | – | – | – | – |
| findoc-c1200 | no_exhibits | hybrid | 0.6 | – | – | – | – |

## Tech Stack

- **Document Parsing:** PyMuPDF (text extraction + table detection)
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Metadata Enrichment:** 10-K section inference from `Item N.` headings, with TOC, signature-page, and exhibit detection (`metadata.py`)
- **Embedding:** BAAI/bge-small-en-v1.5 (384-dim)
- **Search Engine:** OpenSearch 2.17 — BM25 + k-NN (HNSW, Lucene engine) + native hybrid query (`opensearch_store.py`)
- **Vector Store (original):** ChromaDB (cosine similarity)
- **Sparse Retrieval (original):** BM25 (custom implementation)
- **LLM:** OpenAI GPT-4o-mini
- **Frontend:** Streamlit
- **Evaluation:** Custom pipeline (keyword accuracy, faithfulness, relevance)

## Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/findoc-qa.git
cd findoc-qa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a 10-K PDF into `data/pdfs/` and set your OpenAI key:
```bash
export OPENAI_API_KEY=sk-...
```

### OpenSearch backend
```bash
# 1. Start a local OpenSearch node (Docker Desktop must be running)
docker compose up -d
curl http://localhost:9200            # should print cluster info

# 2. Build the indexes (chunk size 800 reuses data/chunks.json; 1200 re-chunks the PDF)
python build_opensearch_index.py
python build_opensearch_index.py --chunk-size 1200 --chunk-overlap 300

# 3. Evaluate: BM25 / dense / hybrid (7 weights), with and without the exhibit filter
python run_opensearch_eval.py                 # retrieval metrics only, no API cost
python run_opensearch_eval.py --llm           # adds end-to-end accuracy with gpt-4o-mini

# 4. Launch the demo and pick "OpenSearch" in the sidebar
streamlit run app.py

# Unit tests (no OpenSearch node needed)
python -m unittest discover -s tests -v
```

### ChromaDB backend (original)
```bash
python findoc_qa.py        # builds data/chroma_db and runs the original experiments
streamlit run app.py       # pick "ChromaDB (original)" in the sidebar
```

## Project Structure
```
findoc-qa/
├── findoc_qa.py          # Original pipeline: parse → chunk → embed (ChromaDB) → evaluate
├── app.py                # Streamlit demo frontend (OpenSearch or ChromaDB backend)
├── opensearch_store.py   # OpenSearch index mapping, bulk indexing, BM25 / k-NN / hybrid search
├── metadata.py           # Metadata enrichment: 10-K section inference, document fields
├── build_opensearch_index.py  # Chunk → enrich → embed → index
├── run_opensearch_eval.py     # Retrieval + end-to-end evaluation on OpenSearch
├── eval_data.py          # Shared 12-question ground truth and metrics
├── docker-compose.yml    # Single-node OpenSearch for local development
├── tests/                # Unit tests with a fake OpenSearch client
├── run_ablation.py       # Ablation experiments (chunk_size × top_k)
├── requirements.txt
├── data/
│   ├── chunks.json       # Pre-computed chunks with metadata
│   └── pdfs/             # Place 10-K PDFs here (not tracked by git)
└── eval/
    ├── evaluation_results.json
    ├── ablation_results.json
    ├── hybrid_vs_dense.json
    └── weight_ablation.json
```

## Evaluation Methodology

- **12-question ground truth test set** covering numerical (8) and factual (4) queries
- **Keyword accuracy:** Does the answer contain expected key figures/terms?
- **Faithfulness:** Are numerical claims in the answer traceable to retrieved sources?
- **Relevance:** Average cosine similarity of top-K retrieved chunks to query
- **9-configuration ablation** over chunk_size (500/800/1200) × top_k (3/5/8)
- **7-configuration weight ablation** for hybrid search dense_weight (0.3–0.9)
- **Context hit rate (OpenSearch eval):** Does any expected keyword appear in the retrieved chunks? This is a free, retrieval-only upper bound on answer accuracy.
- **Latency:** Mean wall-clock search time per query
