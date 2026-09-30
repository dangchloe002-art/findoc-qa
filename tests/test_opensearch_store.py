"""
Unit tests for the OpenSearch backend and metadata enrichment.
These tests use a fake client, so no running OpenSearch node is needed:
    python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opensearch_store as oss  # noqa: E402
from metadata import EXHIBIT_DOCS, FRONT_MATTER, SIGNATURES, enrich_chunks, find_toc_pages, infer_sections  # noqa: E402


class FakeIndices:
    def __init__(self):
        self.existing = set()
        self.created = {}

    def exists(self, index):
        return index in self.existing

    def delete(self, index):
        self.existing.discard(index)

    def create(self, index, body):
        self.existing.add(index)
        self.created[index] = body


class FakeTransport:
    def __init__(self):
        self.requests = []

    def perform_request(self, method, url, body=None):
        self.requests.append((method, url, body))
        return {"acknowledged": True}


class FakeClient:
    def __init__(self, response):
        self.indices = FakeIndices()
        self.transport = FakeTransport()
        self.response = response
        self.searches = []

    def search(self, index, body, params=None):
        self.searches.append({"index": index, "body": body, "params": params})
        return self.response


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=False, **kwargs):
        return [[1.0, 0.0, 0.0] for _ in texts]


def hit(chunk_id, score, page=1, section="Risk Factors"):
    return {"_id": chunk_id, "_score": score,
            "_source": {"chunk_id": chunk_id, "text": f"text of {chunk_id}",
                        "page_num": page, "chunk_index": 0, "has_table": False,
                        "section": section}}


class TestIndexBody(unittest.TestCase):
    def test_mapping_has_text_vector_and_metadata(self):
        props = oss.index_body(384)["mappings"]["properties"]
        self.assertEqual(props["text"]["analyzer"], "english")
        self.assertEqual(props["embedding"]["type"], "knn_vector")
        self.assertEqual(props["embedding"]["dimension"], 384)
        self.assertEqual(props["embedding"]["method"]["space_type"], "cosinesimil")
        self.assertEqual(props["section"]["type"], "keyword")
        self.assertTrue(oss.index_body()["settings"]["index"]["knn"])

    def test_create_index_recreates(self):
        client = FakeClient({})
        client.indices.existing.add("findoc")
        oss.create_index(client, "findoc", dim=8)
        self.assertEqual(client.indices.created["findoc"]["mappings"]["properties"]
                         ["embedding"]["dimension"], 8)


class TestDocuments(unittest.TestCase):
    def test_build_documents_and_bulk_ids(self):
        chunks = [{"id": "p1_c0", "text": "a", "page_num": 1, "chunk_index": 0,
                   "has_table": True, "section": "Business", "fiscal_year": 2024}]
        docs = oss.build_documents(chunks, [[0.1, 0.2]], chunk_size=800, chunk_overlap=200)
        self.assertEqual(docs[0]["embedding"], [0.1, 0.2])
        self.assertEqual(docs[0]["section"], "Business")
        self.assertEqual(docs[0]["chunk_size"], 800)
        actions = oss.bulk_actions(docs, "idx")
        self.assertEqual(actions[0]["_id"], "p1_c0")
        self.assertEqual(actions[0]["_index"], "idx")

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            oss.build_documents([{"id": "x"}], [])


class TestPipeline(unittest.TestCase):
    def test_weights_follow_query_order(self):
        body = oss.pipeline_body(0.6)
        proc = body["phase_results_processors"][0]["normalization-processor"]
        self.assertEqual(proc["normalization"]["technique"], "min_max")
        self.assertEqual(proc["combination"]["parameters"]["weights"], [0.4, 0.6])

    def test_weights_sum_to_one(self):
        for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            weights = oss.pipeline_body(w)["phase_results_processors"][0][
                "normalization-processor"]["combination"]["parameters"]["weights"]
            self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_invalid_weight(self):
        with self.assertRaises(ValueError):
            oss.pipeline_body(1.5)

    def test_pipeline_created_once_per_client(self):
        client = FakeClient({})
        oss.ensure_hybrid_pipeline(client, 0.7)
        oss.ensure_hybrid_pipeline(client, 0.7)
        self.assertEqual(len(client.transport.requests), 1)
        method, url, _ = client.transport.requests[0]
        self.assertEqual((method, url), ("PUT", "/_search/pipeline/findoc-hybrid-w070"))


class TestQueries(unittest.TestCase):
    def test_hybrid_query_structure(self):
        body = oss.hybrid_query("net income", [0.1, 0.2], top_k=5)
        subs = body["query"]["hybrid"]["queries"]
        self.assertEqual(subs[0], {"match": {"text": {"query": "net income"}}})
        self.assertEqual(subs[1]["knn"]["embedding"]["k"], 15)  # 3 * top_k candidates
        self.assertEqual(body["size"], 5)

    def test_filters_apply_to_both_subqueries(self):
        filters = oss.metadata_filter(sections=["Risk Factors"], has_table=False)
        body = oss.hybrid_query("risk", [0.1], top_k=3, filters=filters)
        bm25, knn = body["query"]["hybrid"]["queries"]
        self.assertEqual(bm25["bool"]["filter"], filters)
        self.assertEqual(knn["knn"]["embedding"]["filter"]["bool"]["filter"], filters)

    def test_metadata_filter_clauses(self):
        f = oss.metadata_filter(sections=["A"], has_table=True, page_range=(10, 20),
                                fiscal_year=2024)
        self.assertIn({"terms": {"section": ["A"]}}, f)
        self.assertIn({"term": {"has_table": True}}, f)
        self.assertIn({"range": {"page_num": {"gte": 10, "lte": 20}}}, f)
        self.assertIn({"term": {"fiscal_year": 2024}}, f)
        self.assertEqual(oss.metadata_filter(), [])

    def test_exclude_sections(self):
        f = oss.metadata_filter(exclude_sections=["Attached Exhibit Documents"])
        self.assertEqual(f, [{"bool": {"must_not": [
            {"terms": {"section": ["Attached Exhibit Documents"]}}]}}])


class TestResultConversion(unittest.TestCase):
    def test_dense_scores_become_cosine_distances(self):
        # Lucene cosinesimil score = (1 + cos) / 2, so 0.9 -> cos 0.8 -> distance 0.2
        out = oss.to_chroma_format({"hits": {"hits": [hit("a", 0.9)]}}, "dense")
        self.assertAlmostEqual(out["distances"][0][0], 0.2)

    def test_bm25_scores_normalized_by_max(self):
        out = oss.to_chroma_format({"hits": {"hits": [hit("a", 10.0), hit("b", 5.0)]}}, "bm25")
        self.assertEqual(out["distances"][0], [0.0, 0.5])

    def test_hybrid_layout_matches_chroma(self):
        out = oss.to_chroma_format({"hits": {"hits": [hit("a", 0.75, page=38)]}}, "hybrid")
        self.assertEqual(set(out), {"ids", "documents", "distances", "metadatas", "scores"})
        self.assertEqual(out["ids"], [["a"]])
        self.assertAlmostEqual(out["distances"][0][0], 0.25)
        self.assertEqual(out["metadatas"][0][0]["page_num"], 38)
        self.assertNotIn("chunk_id", out["metadatas"][0][0])

    def test_empty_response(self):
        out = oss.to_chroma_format({"hits": {"hits": []}}, "hybrid")
        self.assertEqual(out["ids"], [[]])


class TestSearch(unittest.TestCase):
    def test_hybrid_search_uses_pipeline(self):
        client = FakeClient({"hits": {"hits": [hit("a", 0.8)]}})
        out = oss.search(client, "revenue", FakeEmbedder(), mode="hybrid",
                         top_k=3, dense_weight=0.6, index="idx")
        call = client.searches[0]
        self.assertEqual(call["index"], "idx")
        self.assertEqual(call["params"], {"search_pipeline": "findoc-hybrid-w060"})
        self.assertIn("hybrid", call["body"]["query"])
        self.assertEqual(out["documents"], [["text of a"]])

    def test_bm25_needs_no_embedder(self):
        client = FakeClient({"hits": {"hits": [hit("a", 3.0)]}})
        oss.search(client, "revenue", None, mode="bm25")
        self.assertIn("match", client.searches[0]["body"]["query"])

    def test_dense_without_embedder_raises(self):
        with self.assertRaises(ValueError):
            oss.search(FakeClient({}), "q", None, mode="dense")


class TestMetadata(unittest.TestCase):
    def test_sections_follow_headings(self):
        chunks = [
            {"id": "p1_c0", "page_num": 1, "chunk_index": 0, "text": "Cover page"},
            {"id": "p3_c0", "page_num": 3, "chunk_index": 0,
             "text": "Item 1.\nBusiness\nItem 1A.\nRisk Factors\nItem 2.\nItem 3.\nItem 7.\n"},
            {"id": "p4_c0", "page_num": 4, "chunk_index": 0, "text": "Item 1.\nBusiness\nApple designs..."},
            {"id": "p5_c0", "page_num": 5, "chunk_index": 0, "text": "More business text"},
            {"id": "p8_c0", "page_num": 8, "chunk_index": 0, "text": "Item 1A.\nRisk Factors\n..."},
            {"id": "p8_c1", "page_num": 8, "chunk_index": 1, "text": "Supply chain risk"},
        ]
        self.assertEqual(find_toc_pages(chunks), {3})
        tagged = {c["id"]: c for c in infer_sections(chunks)}
        self.assertEqual(tagged["p1_c0"]["section"], FRONT_MATTER)
        self.assertEqual(tagged["p3_c0"]["section"], FRONT_MATTER)  # TOC page
        self.assertEqual(tagged["p5_c0"]["section"], "Business")
        self.assertEqual(tagged["p8_c1"]["section"], "Risk Factors")
        self.assertEqual(tagged["p8_c1"]["section_item"], "1A")

    def test_real_apple_chunks(self):
        path = ROOT / "data" / "chunks.json"
        if not path.exists():
            self.skipTest("data/chunks.json not found")
        chunks = json.loads(path.read_text(encoding="utf-8"))
        enriched = enrich_chunks(chunks, "Apple Inc.", 2024, "apple_10k_2024.pdf")
        by_page = {}
        for c in enriched:
            by_page.setdefault(c["page_num"], c["section"])
        self.assertEqual(len(enriched), len(chunks))
        self.assertEqual(by_page[10], "Risk Factors")
        self.assertEqual(by_page[26], "Management's Discussion and Analysis (MD&A)")
        self.assertEqual(by_page[40], "Financial Statements and Supplementary Data")
        self.assertEqual(by_page[60], SIGNATURES)
        self.assertEqual(by_page[100], EXHIBIT_DOCS)
        self.assertTrue(all(c["company"] == "Apple Inc." for c in enriched))

    def test_signatures_and_exhibits_do_not_inherit_last_item(self):
        chunks = [
            {"id": "a", "page_num": 1, "chunk_index": 0, "text": "Item 16.\nForm 10-K Summary\nNone."},
            {"id": "b", "page_num": 2, "chunk_index": 0, "text": "SIGNATURES\nPursuant to ..."},
            {"id": "c", "page_num": 3, "chunk_index": 0, "text": "Exhibit 4.1\nDESCRIPTION OF SECURITIES"},
            {"id": "d", "page_num": 4, "chunk_index": 0, "text": "Item 1.\nnot a real heading here"},
        ]
        tagged = {c["id"]: c["section"] for c in infer_sections(chunks)}
        self.assertEqual(tagged, {"a": "Form 10-K Summary", "b": SIGNATURES,
                                  "c": EXHIBIT_DOCS, "d": EXHIBIT_DOCS})


if __name__ == "__main__":
    unittest.main()
