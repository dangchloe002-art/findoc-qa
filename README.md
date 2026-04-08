# 📄 FinDoc QA — Financial Document Question Answering

An end-to-end RAG (Retrieval-Augmented Generation) system for financial document QA. Upload a 10-K filing and ask questions — the system retrieves relevant passages using hybrid search (BM25 + dense embeddings) and generates citation-grounded answers.

## Architecture
```
PDF → PyMuPDF Parser → Recursive Chunking → BGE-small Embedding → ChromaDB
                                                                      ↓
User Question → Hybrid Search (BM25 + Dense) → Top-K Retrieval → GPT-4o-mini → Answer with [Source N] Citations
```

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

## Tech Stack

- **Document Parsing:** PyMuPDF (text extraction + table detection)
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Embedding:** BAAI/bge-small-en-v1.5 (384-dim)
- **Vector Store:** ChromaDB (cosine similarity)
- **Sparse Retrieval:** BM25 (custom implementation)
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

Download a 10-K PDF into `data/pdfs/`, then:
```bash
# Build index
python findoc_qa.py

# Launch demo
streamlit run app.py
```

Set your OpenAI API key in `app.py` and `findoc_qa.py` before running.

## Project Structure
```
findoc-qa/
├── findoc_qa.py          # Core pipeline: parse → chunk → embed → search → evaluate
├── app.py                # Streamlit demo frontend
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
