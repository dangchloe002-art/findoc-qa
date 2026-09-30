"""
FinDoc QA MCP server.

Exposes the OpenSearch index of the 10-K as Model Context Protocol tools,
so an AI agent (e.g. Claude Desktop) can search the filing, filter by
10-K section, and pull exact chunks to cite. The server only retrieves;
the agent does the reasoning and writes the answer.

Tools:
  - search_filing  BM25 / dense / hybrid search with metadata filters
  - get_chunk      one chunk by id, with full text and metadata
  - get_page       every chunk on one page, in reading order
  - list_sections  10-K sections in the index, with chunk counts

Run (stdio transport, as Claude Desktop expects):
    python mcp_server.py

Environment variables:
    FINDOC_INDEX      OpenSearch index to query (default: findoc-c1200)
    OPENSEARCH_HOST   default: localhost
    OPENSEARCH_PORT   default: 9200
"""

import logging
import os
import sys

import opensearch_store as oss
from metadata import EXHIBIT_DOCS, TENK_ITEMS

# stdout carries the MCP protocol, so all logging goes to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s findoc-mcp %(levelname)s %(message)s")
log = logging.getLogger("findoc-mcp")

DEFAULT_INDEX = os.environ.get("FINDOC_INDEX", "findoc-c1200")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOP_K = 20
VALID_MODES = ("hybrid", "dense", "bm25")


class FinDocTools:
    """
    Tool logic, kept separate from the MCP wiring so it can be unit-tested
    with a fake OpenSearch client and no MCP SDK installed.
    """

    def __init__(self, client, index: str = DEFAULT_INDEX, embed_model=None):
        self.client = client
        self.index = index
        self._embed_model = embed_model

    @property
    def embed_model(self):
        # Load the embedding model on first use, so BM25-only calls stay fast.
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model %s", EMBED_MODEL)
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    # ---------------- helpers ----------------
    @staticmethod
    def _citation(meta: dict) -> str:
        section = meta.get("section", "")
        return f"p.{meta.get('page_num')}" + (f", {section}" if section else "")

    def _valid_sections(self) -> list[str]:
        return [s["section"] for s in self.list_sections()["sections"]]

    # ---------------- tools ----------------
    def search_filing(self, query: str, mode: str = "hybrid", top_k: int = 5,
                      dense_weight: float = 0.6, sections: list[str] | None = None,
                      tables_only: bool = False, exclude_exhibits: bool = False) -> dict:
        if not query or not query.strip():
            return {"error": "query must not be empty"}
        if mode not in VALID_MODES:
            return {"error": f"mode must be one of {list(VALID_MODES)}"}
        if not 0.0 <= dense_weight <= 1.0:
            return {"error": "dense_weight must be between 0 and 1"}
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        if sections:
            valid = set(self._valid_sections())
            unknown = [s for s in sections if s not in valid]
            if unknown:
                return {"error": f"unknown sections: {unknown}",
                        "valid_sections": sorted(valid)}

        filters = oss.metadata_filter(
            sections=sections or None,
            exclude_sections=[EXHIBIT_DOCS] if exclude_exhibits else None,
            has_table=True if tables_only else None)
        res = oss.search(self.client, query,
                         None if mode == "bm25" else self.embed_model,
                         mode=mode, top_k=top_k, dense_weight=dense_weight,
                         index=self.index, filters=filters)

        results = []
        for rank, (cid, text, meta, score) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["scores"][0]), 1):
            results.append({
                "rank": rank,
                "chunk_id": cid,
                "citation": self._citation(meta),
                "page": meta.get("page_num"),
                "section": meta.get("section", ""),
                "has_table": meta.get("has_table", False),
                "score": round(float(score), 4),
                "text": text,
            })
        return {"query": query, "mode": mode, "index": self.index,
                "num_results": len(results), "results": results}

    def get_chunk(self, chunk_id: str) -> dict:
        try:
            doc = self.client.get(index=self.index, id=chunk_id,
                                  _source_excludes=["embedding"])
        except Exception as e:  # NotFoundError or connection errors
            return {"error": f"chunk {chunk_id!r} not found ({type(e).__name__})"}
        src = doc["_source"]
        meta = {k: src[k] for k in oss.METADATA_FIELDS if k in src}
        return {"chunk_id": chunk_id, "citation": self._citation(meta),
                "text": src.get("text", ""), "metadata": meta}

    def get_page(self, page_num: int) -> dict:
        body = {"size": 100, "_source": oss.SOURCE_FIELDS,
                "query": {"term": {"page_num": int(page_num)}},
                "sort": [{"chunk_index": "asc"}]}
        hits = self.client.search(index=self.index, body=body)["hits"]["hits"]
        if not hits:
            return {"error": f"no chunks on page {page_num}"}
        chunks = [{"chunk_id": h["_source"]["chunk_id"],
                   "chunk_index": h["_source"]["chunk_index"],
                   "text": h["_source"]["text"]} for h in hits]
        first = hits[0]["_source"]
        return {"page": int(page_num), "section": first.get("section", ""),
                "has_table": first.get("has_table", False),
                "num_chunks": len(chunks), "chunks": chunks}

    def list_sections(self) -> dict:
        body = {"size": 0, "aggs": {
            "sections": {"terms": {"field": "section", "size": 100},
                         "aggs": {"first_page": {"min": {"field": "page_num"}},
                                  "last_page": {"max": {"field": "page_num"}}}}}}
        buckets = self.client.search(index=self.index, body=body)["aggregations"]["sections"]["buckets"]
        item_of = {name: code for code, name in TENK_ITEMS.items()}
        sections = [{"section": b["key"],
                     "item": item_of.get(b["key"], ""),
                     "chunks": b["doc_count"],
                     "pages": f"{int(b['first_page']['value'])}-{int(b['last_page']['value'])}"}
                    for b in buckets]
        sections.sort(key=lambda s: int(s["pages"].split("-")[0]))
        return {"index": self.index, "sections": sections}


def build_server(tools: FinDocTools):
    """Register the tools on a FastMCP server."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("findoc-qa")

    @mcp.tool()
    def search_filing(query: str, mode: str = "hybrid", top_k: int = 5,
                      dense_weight: float = 0.6, sections: list[str] | None = None,
                      tables_only: bool = False, exclude_exhibits: bool = False) -> dict:
        """Search Apple's FY2024 Form 10-K and return the most relevant text chunks with citations.

        Use this first for any question about the filing. Cite results by their
        'citation' field (page and 10-K section).

        Args:
            query: Natural-language question or keywords.
            mode: "hybrid" (BM25 + vector, best default), "dense" (semantic), or "bm25" (exact keywords and numbers).
            top_k: Number of chunks to return (1-20).
            dense_weight: Weight of vector search in hybrid mode (0-1). BM25 gets 1 - dense_weight.
            sections: Only search these 10-K sections. Call list_sections for valid names.
            tables_only: Only return chunks from pages that contain tables (useful for financial figures).
            exclude_exhibits: Skip the legal exhibit documents attached after the signature page.
        """
        return tools.search_filing(query, mode, top_k, dense_weight, sections,
                                   tables_only, exclude_exhibits)

    @mcp.tool()
    def get_chunk(chunk_id: str) -> dict:
        """Return one chunk's full text and metadata by its chunk_id (e.g. "p38_c0"). Use it to verify a citation."""
        return tools.get_chunk(chunk_id)

    @mcp.tool()
    def get_page(page_num: int) -> dict:
        """Return every chunk on one page of the filing, in reading order. Use it to read the context around a search hit."""
        return tools.get_page(page_num)

    @mcp.tool()
    def list_sections() -> dict:
        """List the 10-K sections in the index with their Item number, chunk count, and page range."""
        return tools.list_sections()

    return mcp


def main():
    host = os.environ.get("OPENSEARCH_HOST", "localhost")
    port = int(os.environ.get("OPENSEARCH_PORT", "9200"))
    client = oss.get_client(host, port)
    log.info("Serving index %s from %s:%s", DEFAULT_INDEX, host, port)
    build_server(FinDocTools(client, DEFAULT_INDEX)).run()  # stdio transport


if __name__ == "__main__":
    main()
