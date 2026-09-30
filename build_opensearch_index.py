"""
Build the FinDoc QA OpenSearch index.

Pipeline: load chunks (from data/chunks.json, or re-chunk the PDF with a
given chunk size) -> enrich metadata (10-K section, company, fiscal year)
-> embed with BGE-small -> bulk-index into OpenSearch.

Usage:
    docker compose up -d                        # start OpenSearch
    python build_opensearch_index.py            # index data/chunks.json (chunk=800)
    python build_opensearch_index.py --chunk-size 1200 --chunk-overlap 300
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from metadata import enrich_chunks
import opensearch_store as oss

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def parse_pdf(pdf_path: str) -> list[dict]:
    """Extract text per page and flag pages that contain tables (same as findoc_qa.py)."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        tables = page.find_tables()
        pages.append({
            "page_num": i + 1,
            "text": text.strip(),
            "has_table": len(tables.tables) > 0 if tables else False,
        })
    doc.close()
    return pages


def chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Recursive character chunking with page metadata (same as findoc_qa.py)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for page in pages:
        if not page["text"]:
            continue
        for j, text in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "id": f"p{page['page_num']}_c{j}",
                "text": text,
                "page_num": page["page_num"],
                "has_table": page["has_table"],
                "chunk_index": j,
            })
    return chunks


def index_name_for(chunk_size: int) -> str:
    return f"{oss.DEFAULT_INDEX}-c{chunk_size}"


def build(chunk_size: int = 800, chunk_overlap: int = 200,
          pdf: str = "data/pdfs/apple_10k_2024.pdf",
          chunks_path: str = "data/chunks.json",
          index: str | None = None, host: str = "localhost", port: int = 9200,
          company: str = "Apple Inc.", fiscal_year: int = 2024,
          embed_model=None, client=None) -> str:
    """Build one index and return its name."""
    index = index or index_name_for(chunk_size)

    # 1. Chunks: reuse the saved 800/200 chunks, or re-chunk the PDF
    if chunk_size == 800 and chunk_overlap == 200 and Path(chunks_path).exists():
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    else:
        chunks = chunk_pages(parse_pdf(pdf), chunk_size, chunk_overlap)
        print(f"Re-chunked {pdf}: {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # 2. Metadata enrichment
    chunks = enrich_chunks(chunks, company=company, fiscal_year=fiscal_year,
                           source_file=Path(pdf).name)
    sections = Counter(c["section"] for c in chunks)
    print("Chunks per 10-K section:")
    for name, n in sections.most_common():
        print(f"   {n:4d}  {name}")
    out = Path(f"data/chunks_enriched_c{chunk_size}.json")
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved enriched chunks -> {out}")

    # 3. Embeddings (normalized, so cosine = dot product)
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL)
    start = time.time()
    embeddings = embed_model.encode([c["text"] for c in chunks], batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedded {len(chunks)} chunks in {time.time() - start:.1f}s")

    # 4. Index
    client = client or oss.get_client(host, port)
    oss.create_index(client, index=index, dim=len(embeddings[0]), recreate=True)
    n = oss.index_chunks(client, chunks, embeddings, index=index,
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Indexed {n} documents -> OpenSearch index '{index}'")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--pdf", default="data/pdfs/apple_10k_2024.pdf")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--index", default=None, help="default: findoc-c<chunk_size>")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9200)
    args = ap.parse_args()
    build(args.chunk_size, args.chunk_overlap, args.pdf, args.chunks,
          args.index, args.host, args.port)
