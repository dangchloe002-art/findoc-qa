"""
Evaluate OpenSearch retrieval (BM25 / dense / native hybrid) on the
12-question ground-truth set.

Two levels of evaluation:
  1. Retrieval only (free, no API calls):
       - context hit rate: share of questions where at least one expected
         keyword appears in the retrieved chunks (an upper bound on
         answer accuracy)
       - relevance: mean cosine similarity between query and retrieved chunks
       - latency: mean search time per query
  2. End-to-end RAG (--llm, needs OPENAI_API_KEY): the same gpt-4o-mini
     prompt as findoc_qa.py, scored with keyword accuracy and faithfulness.

Usage:
    python run_opensearch_eval.py --build                  # build indexes, then evaluate
    python run_opensearch_eval.py --chunk-sizes 800 1200 --llm
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import opensearch_store as oss
from build_opensearch_index import EMBED_MODEL, build, index_name_for
from metadata import EXHIBIT_DOCS
from eval_data import (GROUND_TRUTH, SYSTEM_PROMPT, build_context,
                       eval_faithfulness, eval_keyword_hit, eval_relevance)

OVERLAP = {500: 100, 800: 200, 1200: 300}


class EmbeddingCache:
    """Encode texts once and reuse the (normalized) vectors."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def get(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self.cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, batch_size=32)
            self.cache.update(zip(missing, vecs))
        return np.array([self.cache[t] for t in texts])


def llm_answer(openai_client, question: str, results: dict) -> tuple[str, int]:
    context = build_context(results["documents"][0], results["metadatas"][0],
                            results["distances"][0])
    user_prompt = (f"Context from Apple 10-K filing:\n\n{context}\n\n---\n\n"
                   f"Question: {question}\n\nAnswer with citations:")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}],
        temperature=0, max_tokens=800)
    return resp.choices[0].message.content, resp.usage.total_tokens


def evaluate_config(client, index, embed_model, cache, mode, top_k,
                    dense_weight=None, openai_client=None, filters=None,
                    filter_label="none") -> dict:
    per_q = []
    for item in GROUND_TRUTH:
        q = item["question"]
        t0 = time.perf_counter()
        res = oss.search(client, q, embed_model, mode=mode, top_k=top_k,
                         dense_weight=dense_weight if dense_weight is not None else 0.6,
                         index=index, filters=filters)
        latency_ms = (time.perf_counter() - t0) * 1000

        docs = res["documents"][0]
        q_vec = cache.get([q])[0]
        sims = (cache.get(docs) @ q_vec).tolist() if docs else []
        row = {
            "question": q,
            "category": item["category"],
            "context_hit": eval_keyword_hit(" ".join(docs), item["keywords"]),
            "relevance": eval_relevance(sims),
            "latency_ms": latency_ms,
            "pages": [m["page_num"] for m in res["metadatas"][0]],
            "sections": [m.get("section", "") for m in res["metadatas"][0]],
        }
        if openai_client is not None:
            answer, tokens = llm_answer(openai_client, q, res)
            row.update({"answer": answer, "tokens": tokens,
                        "keyword_hit": eval_keyword_hit(answer, item["keywords"]),
                        "faithfulness": eval_faithfulness(answer, docs)})
        per_q.append(row)

    n = len(per_q)
    summary = {
        "index": index, "mode": mode, "top_k": top_k, "dense_weight": dense_weight,
        "filter": filter_label,
        "context_hit_rate": sum(r["context_hit"] for r in per_q) / n,
        "relevance": sum(r["relevance"] for r in per_q) / n,
        "latency_ms": sum(r["latency_ms"] for r in per_q) / n,
    }
    if openai_client is not None:
        summary["accuracy"] = sum(r["keyword_hit"] for r in per_q) / n
        summary["faithfulness"] = sum(r["faithfulness"] for r in per_q) / n
        summary["tokens"] = sum(r["tokens"] for r in per_q)
    summary["details"] = per_q
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[800, 1200])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--build", action="store_true", help="(re)build the indexes first")
    ap.add_argument("--llm", action="store_true", help="also run end-to-end RAG with gpt-4o-mini")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--out", default="eval/opensearch_eval.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)
    cache = EmbeddingCache(embed_model)
    client = oss.get_client(args.host, args.port)

    openai_client = None
    if args.llm:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--llm needs OPENAI_API_KEY in the environment")
        openai_client = OpenAI()

    results = []
    for cs in args.chunk_sizes:
        index = index_name_for(cs)
        if args.build or not client.indices.exists(index=index):
            build(cs, OVERLAP.get(cs, cs // 4), index=index, host=args.host,
                  port=args.port, embed_model=embed_model, client=client)

        # Every retrieval mode runs twice: on the whole filing, and with the
        # attached exhibit documents (indentures, stock plans) filtered out
        # through the section metadata.
        configs = [("bm25", None), ("dense", None)] + [("hybrid", w) for w in args.weights]
        filter_sets = [("none", None),
                       ("no_exhibits", oss.metadata_filter(exclude_sections=[EXHIBIT_DOCS]))]
        for flabel, filters in filter_sets:
            for mode, w in configs:
                label = f"{index} | {mode}" + (f" w={w:.1f}" if w is not None else "") + f" | filter={flabel}"
                print(f"Running {label} ...")
                results.append(evaluate_config(client, index, embed_model, cache, mode,
                                               args.top_k, w, openai_client,
                                               filters=filters, filter_label=flabel))

    # Summary table
    has_llm = openai_client is not None
    header = f"{'index':<12} {'filter':<12} {'mode':<7} {'w':>4} {'ctx_hit':>8} {'rel':>6} {'ms':>6}"
    if has_llm:
        header += f" {'acc':>6} {'faith':>6}"
    print("\n" + header + "\n" + "-" * len(header))
    for r in results:
        w = f"{r['dense_weight']:.1f}" if r["dense_weight"] is not None else "-"
        line = (f"{r['index']:<12} {r['filter']:<12} {r['mode']:<7} {w:>4} {r['context_hit_rate']:>7.1%} "
                f"{r['relevance']:>6.3f} {r['latency_ms']:>6.1f}")
        if has_llm:
            line += f" {r['accuracy']:>5.1%} {r['faithfulness']:>6.3f}"
        print(line)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
