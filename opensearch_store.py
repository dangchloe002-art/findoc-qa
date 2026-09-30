"""
OpenSearch retrieval backend for FinDoc QA.

One index holds three kinds of fields for every chunk:
  - `text`       full-text field (English analyzer) for BM25 keyword search
  - `embedding`  knn_vector field (HNSW, cosine) for dense vector search
  - metadata     page, table flag, 10-K section, company, fiscal year, ...

Three search modes are supported:
  - "bm25"    keyword search on `text`
  - "dense"   approximate k-NN search on `embedding`
  - "hybrid"  OpenSearch's native `hybrid` query. A search pipeline
              min-max normalizes the BM25 and k-NN scores and combines
              them with a weighted arithmetic mean. This is the same
              fusion idea as the original custom BM25 + ChromaDB code,
              but it runs inside the search engine.

All search functions return the same dict layout that ChromaDB's
`collection.query()` returns ("ids", "documents", "distances",
"metadatas"), so the existing `rag_answer()` code works unchanged.

The OpenSearch client is passed in as an argument. This keeps the module
testable with a fake client and free of global state.
"""

from __future__ import annotations

DEFAULT_INDEX = "findoc"
EMBED_DIM = 384  # BAAI/bge-small-en-v1.5

METADATA_FIELDS = [
    "chunk_id", "page_num", "chunk_index", "has_table",
    "section_item", "section", "company", "form_type",
    "fiscal_year", "source_file",
]


# ============================================================
# Client
# ============================================================
def get_client(host: str = "localhost", port: int = 9200):
    """Create an OpenSearch client for a local, security-disabled node."""
    from opensearchpy import OpenSearch  # imported lazily for testability

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        timeout=60,
    )


# ============================================================
# Index management
# ============================================================
def index_body(dim: int = EMBED_DIM) -> dict:
    """Settings and mappings for the chunk index."""
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "english"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "page_num": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "has_table": {"type": "boolean"},
                "section_item": {"type": "keyword"},
                "section": {"type": "keyword"},
                "company": {"type": "keyword"},
                "form_type": {"type": "keyword"},
                "fiscal_year": {"type": "integer"},
                "source_file": {"type": "keyword"},
                "chunk_size": {"type": "integer"},
                "chunk_overlap": {"type": "integer"},
            }
        },
    }


def create_index(client, index: str = DEFAULT_INDEX, dim: int = EMBED_DIM,
                 recreate: bool = True) -> None:
    """Create the chunk index. Drops an existing index first if recreate=True."""
    if client.indices.exists(index=index):
        if not recreate:
            return
        client.indices.delete(index=index)
    client.indices.create(index=index, body=index_body(dim))


def build_documents(chunks: list[dict], embeddings,
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[dict]:
    """Turn chunks + embeddings into OpenSearch documents."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    docs = []
    for c, emb in zip(chunks, embeddings):
        doc = {
            "chunk_id": c["id"],
            "text": c["text"],
            "embedding": [float(x) for x in emb],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "has_table": bool(c.get("has_table", False)),
        }
        for field in ["section_item", "section", "company", "form_type",
                      "fiscal_year", "source_file"]:
            if field in c:
                doc[field] = c[field]
        if chunk_size is not None:
            doc["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            doc["chunk_overlap"] = chunk_overlap
        docs.append(doc)
    return docs


def bulk_actions(docs: list[dict], index: str) -> list[dict]:
    """Bulk-API actions. The chunk id is the document _id, so re-indexing is idempotent."""
    return [{"_op_type": "index", "_index": index, "_id": d["chunk_id"], "_source": d}
            for d in docs]


def index_chunks(client, chunks: list[dict], embeddings,
                 index: str = DEFAULT_INDEX, chunk_size: int | None = None,
                 chunk_overlap: int | None = None) -> int:
    """Bulk-index chunks and refresh so they are searchable right away."""
    from opensearchpy import helpers  # imported lazily for testability

    docs = build_documents(chunks, embeddings, chunk_size, chunk_overlap)
    success, errors = helpers.bulk(client, bulk_actions(docs, index), raise_on_error=False)
    if errors:
        raise RuntimeError(f"{len(errors)} bulk errors, first: {errors[0]}")
    client.indices.refresh(index=index)
    return success


# ============================================================
# Hybrid search pipeline
# ============================================================
def pipeline_name(dense_weight: float) -> str:
    return f"findoc-hybrid-w{round(dense_weight * 100):03d}"


def pipeline_body(dense_weight: float) -> dict:
    """
    Normalization pipeline for the hybrid query.

    The weights follow the order of sub-queries in `hybrid_query()`:
    [BM25, dense]. Both scores are min-max normalized to [0, 1] first,
    then combined as a weighted arithmetic mean.
    """
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be in [0, 1]")
    w_dense = round(dense_weight, 3)
    w_bm25 = round(1.0 - w_dense, 3)
    return {
        "description": f"FinDoc hybrid: min-max normalization, dense weight {w_dense}",
        "phase_results_processors": [{
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": [w_bm25, w_dense]},
                },
            }
        }],
    }


_CREATED_PIPELINES: set[tuple[int, str]] = set()


def ensure_hybrid_pipeline(client, dense_weight: float, force: bool = False) -> str:
    """
    Create the search pipeline for a dense weight once per client and
    return its name. Later calls with the same weight reuse it.
    """
    name = pipeline_name(dense_weight)
    key = (id(client), name)
    if force or key not in _CREATED_PIPELINES:
        client.transport.perform_request(
            "PUT", f"/_search/pipeline/{name}", body=pipeline_body(dense_weight))
        _CREATED_PIPELINES.add(key)
    return name


# ============================================================
# Query builders
# ============================================================
def metadata_filter(sections: list[str] | None = None,
                    exclude_sections: list[str] | None = None,
                    has_table: bool | None = None,
                    page_range: tuple[int, int] | None = None,
                    fiscal_year: int | None = None) -> list[dict]:
    """Build a list of filter clauses from optional metadata constraints."""
    clauses = []
    if sections:
        clauses.append({"terms": {"section": list(sections)}})
    if exclude_sections:
        clauses.append({"bool": {"must_not": [{"terms": {"section": list(exclude_sections)}}]}})
    if has_table is not None:
        clauses.append({"term": {"has_table": has_table}})
    if page_range is not None:
        lo, hi = page_range
        clauses.append({"range": {"page_num": {"gte": lo, "lte": hi}}})
    if fiscal_year is not None:
        clauses.append({"term": {"fiscal_year": fiscal_year}})
    return clauses


def _bm25_clause(query: str, filters: list[dict]) -> dict:
    match = {"match": {"text": {"query": query}}}
    if not filters:
        return match
    return {"bool": {"must": [match], "filter": filters}}


def _knn_clause(vector, k: int, filters: list[dict]) -> dict:
    knn = {"vector": [float(x) for x in vector], "k": k}
    if filters:
        knn["filter"] = {"bool": {"filter": filters}}
    return {"knn": {"embedding": knn}}


SOURCE_FIELDS = ["text"] + METADATA_FIELDS


def bm25_query(query: str, top_k: int, filters: list[dict] | None = None) -> dict:
    return {"size": top_k, "_source": SOURCE_FIELDS,
            "query": _bm25_clause(query, filters or [])}


def dense_query(vector, top_k: int, filters: list[dict] | None = None,
                num_candidates: int | None = None) -> dict:
    k = max(top_k, num_candidates or top_k)
    return {"size": top_k, "_source": SOURCE_FIELDS,
            "query": _knn_clause(vector, k, filters or [])}


def hybrid_query(query: str, vector, top_k: int, filters: list[dict] | None = None,
                 num_candidates: int | None = None) -> dict:
    """
    Native hybrid query: [BM25 sub-query, k-NN sub-query].
    Each sub-query retrieves `num_candidates` hits (default 3 * top_k,
    matching the candidate pool of the original implementation) before
    the pipeline normalizes and fuses them.
    """
    filters = filters or []
    k = num_candidates or top_k * 3
    return {
        "size": top_k,
        "_source": SOURCE_FIELDS,
        "query": {"hybrid": {"queries": [
            _bm25_clause(query, filters),
            _knn_clause(vector, k, filters),
        ]}},
    }


# ============================================================
# Result conversion
# ============================================================
def lucene_cosine_score_to_similarity(score: float) -> float:
    """Lucene's cosinesimil score is (1 + cos) / 2. Convert it back to cos."""
    return 2.0 * score - 1.0


def to_chroma_format(response: dict, mode: str) -> dict:
    """
    Convert an OpenSearch response to ChromaDB's query() layout.

    'distances' follows ChromaDB's convention (lower = more relevant):
      - dense:  1 - cosine similarity (same scale as ChromaDB)
      - bm25:   1 - score / max score in this result set
      - hybrid: 1 - fused score (already normalized to [0, 1])
    """
    hits = response.get("hits", {}).get("hits", [])
    ids, documents, distances, metadatas, scores = [], [], [], [], []

    max_score = max((h["_score"] or 0.0 for h in hits), default=0.0) or 1.0
    for h in hits:
        src = h["_source"]
        score = h["_score"] or 0.0
        if mode == "dense":
            dist = 1.0 - lucene_cosine_score_to_similarity(score)
        elif mode == "bm25":
            dist = 1.0 - score / max_score
        elif mode == "hybrid":
            dist = 1.0 - score
        else:
            raise ValueError(f"unknown mode: {mode}")

        ids.append(src.get("chunk_id", h.get("_id")))
        documents.append(src["text"])
        distances.append(dist)
        scores.append(score)
        metadatas.append({k: src[k] for k in METADATA_FIELDS if k in src and k != "chunk_id"})

    return {"ids": [ids], "documents": [documents], "distances": [distances],
            "metadatas": [metadatas], "scores": [scores]}


# ============================================================
# Search entry point
# ============================================================
def search(client, query: str, embed_model=None, mode: str = "hybrid",
           top_k: int = 5, dense_weight: float = 0.6, index: str = DEFAULT_INDEX,
           filters: list[dict] | None = None, num_candidates: int | None = None) -> dict:
    """
    Run a BM25, dense, or hybrid search and return ChromaDB-style results.

    embed_model must provide `encode(list[str])` (e.g. SentenceTransformer);
    it is not needed for mode="bm25".
    """
    if mode == "bm25":
        body = bm25_query(query, top_k, filters)
        response = client.search(index=index, body=body)
        return to_chroma_format(response, "bm25")

    if embed_model is None:
        raise ValueError(f"mode={mode!r} needs an embedding model")
    vector = embed_model.encode([query], normalize_embeddings=True)[0]

    if mode == "dense":
        body = dense_query(vector, top_k, filters, num_candidates)
        response = client.search(index=index, body=body)
        return to_chroma_format(response, "dense")

    if mode == "hybrid":
        pipeline = ensure_hybrid_pipeline(client, dense_weight)
        body = hybrid_query(query, vector, top_k, filters, num_candidates)
        response = client.search(index=index, body=body,
                                 params={"search_pipeline": pipeline})
        return to_chroma_format(response, "hybrid")

    raise ValueError(f"unknown mode: {mode}")
