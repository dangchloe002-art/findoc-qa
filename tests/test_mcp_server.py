"""
Unit tests for the MCP tool logic (FinDocTools). A fake OpenSearch client
stands in for the real node, and the MCP SDK is not needed:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_server import FinDocTools  # noqa: E402

SECTIONS_AGG = {"aggregations": {"sections": {"buckets": [
    {"key": "Risk Factors", "doc_count": 119, "first_page": {"value": 8}, "last_page": {"value": 20}},
    {"key": "Business", "doc_count": 28, "first_page": {"value": 4}, "last_page": {"value": 7}},
    {"key": "Financial Statements and Supplementary Data", "doc_count": 106,
     "first_page": {"value": 31}, "last_page": {"value": 53}},
]}}}


def hit(cid, score, page=38, section="Financial Statements and Supplementary Data", idx=0):
    return {"_id": cid, "_score": score, "_source": {
        "chunk_id": cid, "text": f"text of {cid}", "page_num": page,
        "chunk_index": idx, "has_table": True, "section": section}}


class FakeTransport:
    def perform_request(self, method, url, body=None):
        return {"acknowledged": True}


class FakeClient:
    def __init__(self, search_hits=None):
        self.transport = FakeTransport()
        self.search_hits = search_hits or []
        self.searches = []
        self.docs = {"p38_c0": {"_source": hit("p38_c0", 1.0)["_source"]}}

    def search(self, index, body, params=None):
        self.searches.append({"index": index, "body": body, "params": params})
        if "aggs" in body:
            return SECTIONS_AGG
        return {"hits": {"hits": self.search_hits}}

    def get(self, index, id, **kwargs):
        if id not in self.docs:
            raise KeyError(id)
        return self.docs[id]


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=False, **kwargs):
        return [[1.0, 0.0] for _ in texts]


class TestSearchFiling(unittest.TestCase):
    def test_hybrid_results_have_citations(self):
        client = FakeClient([hit("p38_c0", 0.9), hit("p27_c1", 0.5, page=27, section="Business")])
        out = FinDocTools(client, "idx", FakeEmbedder()).search_filing("net income")
        self.assertEqual(out["num_results"], 2)
        first = out["results"][0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["chunk_id"], "p38_c0")
        self.assertEqual(first["citation"], "p.38, Financial Statements and Supplementary Data")
        self.assertEqual(client.searches[0]["params"], {"search_pipeline": "findoc-hybrid-w060"})

    def test_bm25_does_not_load_model(self):
        client = FakeClient([hit("p38_c0", 3.0)])
        tools = FinDocTools(client, "idx", embed_model=None)
        tools.search_filing("net income", mode="bm25")
        self.assertIsNone(tools._embed_model)

    def test_filters_reach_the_query(self):
        client = FakeClient([hit("p38_c0", 0.9)])
        FinDocTools(client, "idx", FakeEmbedder()).search_filing(
            "revenue", sections=["Business"], tables_only=True, exclude_exhibits=True)
        search = [s for s in client.searches if "aggs" not in s["body"]][0]
        bm25_clause = search["body"]["query"]["hybrid"]["queries"][0]
        filters = bm25_clause["bool"]["filter"]
        self.assertIn({"terms": {"section": ["Business"]}}, filters)
        self.assertIn({"term": {"has_table": True}}, filters)
        self.assertTrue(any("must_not" in f.get("bool", {}) for f in filters))

    def test_unknown_section_returns_valid_names(self):
        out = FinDocTools(FakeClient(), "idx", FakeEmbedder()).search_filing(
            "revenue", sections=["Revenue Stuff"])
        self.assertIn("unknown sections", out["error"])
        self.assertIn("Business", out["valid_sections"])

    def test_input_validation(self):
        tools = FinDocTools(FakeClient(), "idx", FakeEmbedder())
        self.assertIn("error", tools.search_filing("  "))
        self.assertIn("error", tools.search_filing("q", mode="fuzzy"))
        self.assertIn("error", tools.search_filing("q", dense_weight=2))

    def test_top_k_is_clamped(self):
        client = FakeClient([])
        FinDocTools(client, "idx", FakeEmbedder()).search_filing("q", mode="bm25", top_k=500)
        self.assertEqual(client.searches[0]["body"]["size"], 20)


class TestOtherTools(unittest.TestCase):
    def test_get_chunk(self):
        out = FinDocTools(FakeClient()).get_chunk("p38_c0")
        self.assertEqual(out["text"], "text of p38_c0")
        self.assertEqual(out["metadata"]["page_num"], 38)
        self.assertNotIn("embedding", out["metadata"])

    def test_get_chunk_missing(self):
        self.assertIn("not found", FinDocTools(FakeClient()).get_chunk("nope")["error"])

    def test_get_page_sorted_query(self):
        client = FakeClient([hit("p38_c0", 1.0), hit("p38_c1", 1.0, idx=1)])
        out = FinDocTools(client).get_page(38)
        self.assertEqual(out["num_chunks"], 2)
        self.assertEqual(client.searches[0]["body"]["query"], {"term": {"page_num": 38}})
        self.assertEqual(client.searches[0]["body"]["sort"], [{"chunk_index": "asc"}])

    def test_get_page_empty(self):
        self.assertIn("error", FinDocTools(FakeClient([])).get_page(999))

    def test_list_sections_sorted_by_page_with_item_codes(self):
        out = FinDocTools(FakeClient()).list_sections()
        names = [s["section"] for s in out["sections"]]
        self.assertEqual(names, ["Business", "Risk Factors",
                                 "Financial Statements and Supplementary Data"])
        self.assertEqual(out["sections"][1]["item"], "1A")
        self.assertEqual(out["sections"][1]["pages"], "8-20")


if __name__ == "__main__":
    unittest.main()
