from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse
import json
import random
import time
from pathlib import Path

from app.evaluation.metrics import mean_ignoring_none, mrr, ndcg_at_k, recall_at_k
from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.models.corpus import EvalQuery
from app.retrieval.retriever import Retriever
from config.loader import load_config

# Spec §14.1: "Comparison table (per strategy, and broken down per language)." Retrieval
# only, no LLM call — independently benchmarkable per spec §8's last bullet. Held-out
# (corpus-excluded) queries are skipped: their relevant passages were deliberately kept
# out of the corpus (spec §5), so recall/nDCG/MRR against them is undefined by
# construction, not a score of 0 — they belong to the abstention curve
# (calibrate_guardrails.py), not this table.


def load_answerable_queries(eval_set_path: Path) -> list[EvalQuery]:
    queries = []
    with open(eval_set_path, encoding="utf-8") as f:
        for line in f:
            eq = EvalQuery(**json.loads(line))
            if not eq.held_out:
                queries.append(eq)
    return queries


def _source_doc_id(chunk_id: str) -> str:
    """chunk_id is "{passage_id}::{strategy}::{index}" (app/chunking/base.py's
    make_chunk) — relevant_passage_ids in the eval set are bare passage_ids. Recall/
    nDCG/MRR need to compare at the passage level (a passage may have split into
    several chunks; a hit on *any* of them counts), so this strips the chunking
    suffix before scoring rather than comparing the two id formats directly, which
    can never match and would silently score every query as a total miss."""
    return chunk_id.rsplit("::", 2)[0]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def evaluate_strategy(
    strategy_name: str, index_dir: Path, encoder: E5Encoder, queries: list[EvalQuery], config
) -> dict:
    strategy_dir = index_dir / strategy_name
    retriever = Retriever(
        encoder,
        DenseIndex.load(strategy_dir / "dense.faiss", ef_search=config.retrieval.ef_search),
        BM25SparseIndex.load(strategy_dir / "sparse"),
        MetadataStore(strategy_dir / "meta.sqlite3"),
        config.retrieval,
    )
    # build_index.py writes per-strategy stats *into* the BUILD_COMPLETE marker file
    # itself (scripts/build_index.py's build_strategy()) rather than a separate file.
    build_summary = {}
    marker_path = strategy_dir / "BUILD_COMPLETE"
    if marker_path.exists():
        build_summary = json.loads(marker_path.read_text(encoding="utf-8"))

    per_language: dict[str, dict[str, list]] = {}
    latencies_ms: list[float] = []

    for eq in queries:
        relevant = set(eq.relevant_passage_ids)
        start = time.perf_counter()
        result = retriever.retrieve(eq.query)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        retrieved_ids = [_source_doc_id(r.chunk_id) for r in result.results]

        bucket = per_language.setdefault(
            eq.language, {"recall@5": [], "ndcg@10": [], "mrr": []}
        )
        bucket["recall@5"].append(recall_at_k(retrieved_ids, relevant, k=5))
        bucket["ndcg@10"].append(ndcg_at_k(retrieved_ids, relevant, k=10))
        bucket["mrr"].append(mrr(retrieved_ids, relevant))

    all_recall = [v for b in per_language.values() for v in b["recall@5"]]
    all_ndcg = [v for b in per_language.values() for v in b["ndcg@10"]]
    all_mrr = [v for b in per_language.values() for v in b["mrr"]]

    return {
        "strategy": strategy_name,
        "n_queries": len(queries),
        "recall@5": mean_ignoring_none(all_recall),
        "ndcg@10": mean_ignoring_none(all_ndcg),
        "mrr": mean_ignoring_none(all_mrr),
        "query_latency_ms_p50": percentile(latencies_ms, 50),
        "query_latency_ms_p70": percentile(latencies_ms, 70),
        "query_latency_ms_p100": percentile(latencies_ms, 100),
        "chunks": build_summary.get("n_chunks"),
        "index_size_mb": build_summary.get("index_size_mb"),
        "build_time_s": build_summary.get("total_build_time_s"),
        "per_language": {
            lang: {
                "n_queries": len(bucket["recall@5"]),
                "recall@5": mean_ignoring_none(bucket["recall@5"]),
                "ndcg@10": mean_ignoring_none(bucket["ndcg@10"]),
                "mrr": mean_ignoring_none(bucket["mrr"]),
            }
            for lang, bucket in sorted(per_language.items())
        },
    }


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_set.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--strategies", default=",".join(config.chunking.strategies))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/retrieval_comparison.json")
    args = parser.parse_args()

    all_queries = load_answerable_queries(Path(args.eval_set))
    rng = random.Random(args.seed)
    rng.shuffle(all_queries)
    queries = all_queries[: args.sample_size]
    print(f"Evaluating on {len(queries)} answerable queries (of {len(all_queries)} available).")

    encoder = E5Encoder()
    results = []
    for strategy_name in args.strategies.split(","):
        strategy_dir = Path(args.index_dir) / strategy_name
        if not (strategy_dir / "BUILD_COMPLETE").exists():
            print(f"[{strategy_name}] no built index, skipping.")
            continue
        print(f"[{strategy_name}] evaluating...", flush=True)
        result = evaluate_strategy(strategy_name, Path(args.index_dir), encoder, queries, config)
        results.append(result)
        print(
            f"  recall@5={result['recall@5']:.3f} ndcg@10={result['ndcg@10']:.3f} "
            f"mrr={result['mrr']:.3f} p50={result['query_latency_ms_p50']:.2f}ms "
            f"p100={result['query_latency_ms_p100']:.2f}ms"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"sample_size": len(queries), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
