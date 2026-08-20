from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.guardrails.retrieval_gate import retrieval_confidence_gate
from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.models.corpus import EvalQuery
from app.retrieval.retriever import Retriever
from config.loader import load_config

# Off-topic queries the corpus was never built to answer (spec §11.4 bucket (a)) —
# hand-authored, not sourced from MS-MARCO-XI, spanning several corpus languages plus
# English. A small, honest sample, not a claim of broad off-topic coverage — same
# limitation already documented for app/guardrails/injection.py and safety.py.
OFF_TOPIC_QUERIES = [
    ("What's the offside rule in football?", "en"),
    ("How do I bake a chocolate cake?", "en"),
    ("Recommend a good sci-fi movie from the 1980s.", "en"),
    ("क्रिकेट में नो-बॉल का नियम क्या है?", "hi"),
    ("आज मौसम कैसा रहेगा?", "hi"),
    ("চকলেট কেক কীভাবে বানাবো?", "bn"),
    ("எனக்கு ஒரு நல்ல திரைப்படத்தை பரிந்துரைக்கவும்.", "ta"),
    ("ఫుట్‌బాల్ నియమాలు ఏమిటి?", "te"),
    ("Best way to learn guitar as a beginner?", "en"),
    ("What is the capital of France?", "en"),
]


@dataclass
class LabeledQuery:
    query: str
    is_answerable: bool
    bucket: str  # "answerable" | "corpus_excluded" | "off_topic"


def load_labeled_queries(eval_set_path: Path, sample_size: int, seed: int) -> list[LabeledQuery]:
    rng = random.Random(seed)
    answerable: list[LabeledQuery] = []
    corpus_excluded: list[LabeledQuery] = []
    with open(eval_set_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            eq = EvalQuery(**row)
            label = LabeledQuery(eq.query, not eq.held_out, "corpus_excluded" if eq.held_out else "answerable")
            (corpus_excluded if eq.held_out else answerable).append(label)

    rng.shuffle(answerable)
    rng.shuffle(corpus_excluded)
    n_each = sample_size // 2
    sampled = answerable[:n_each] + corpus_excluded[:n_each]
    sampled += [LabeledQuery(q, False, "off_topic") for q, _lang in OFF_TOPIC_QUERIES]
    rng.shuffle(sampled)
    return sampled


def evaluate_threshold(scored: list[tuple[LabeledQuery, float | None, list]], tau_abs: float, tau_margin: float) -> dict:
    """`scored` holds (query, top1_fused_score, retrieval_results) per labeled query,
    precomputed once so the sweep below doesn't re-run retrieval per threshold pair —
    spec's tau values are calibrated on this same fixed set of scores; only the
    thresholds move across the grid.
    """
    tp = fp = fn = tn = 0  # positive class: "should answer" (is_answerable)
    for labeled, top1, results in scored:
        allowed = top1 is not None and retrieval_confidence_gate(results, tau_abs, tau_margin).allowed
        if labeled.is_answerable and allowed:
            tp += 1
        elif labeled.is_answerable and not allowed:
            fn += 1
        elif not labeled.is_answerable and allowed:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Abstention-direction metrics (spec's actual ask: how good is the system at
    # *knowing when not to answer*) — positive class flipped to "should abstain".
    abstain_precision = tn / (tn + fn) if (tn + fn) else 0.0
    abstain_recall = tn / (tn + fp) if (tn + fp) else 0.0
    abstain_f1 = (
        2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall)
        if (abstain_precision + abstain_recall)
        else 0.0
    )
    return {
        "tau_abs": tau_abs, "tau_margin": tau_margin,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "answer_precision": precision, "answer_recall": recall, "answer_f1": f1,
        "abstain_precision": abstain_precision, "abstain_recall": abstain_recall, "abstain_f1": abstain_f1,
    }


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_set.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--strategy", default=config.chunking.active_strategy)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/calibration_results.json")
    args = parser.parse_args()

    strategy_dir = Path(args.index_dir) / args.strategy
    print(f"Loading retriever for strategy={args.strategy}...")
    encoder = E5Encoder()
    retriever = Retriever(
        encoder,
        DenseIndex.load(strategy_dir / "dense.faiss", ef_search=config.retrieval.ef_search),
        BM25SparseIndex.load(strategy_dir / "sparse"),
        MetadataStore(strategy_dir / "meta.sqlite3"),
        config.retrieval,
    )

    labeled_queries = load_labeled_queries(Path(args.eval_set), args.sample_size, args.seed)
    n_answerable = sum(1 for q in labeled_queries if q.bucket == "answerable")
    n_excluded = sum(1 for q in labeled_queries if q.bucket == "corpus_excluded")
    n_off_topic = sum(1 for q in labeled_queries if q.bucket == "off_topic")
    print(
        f"Running retrieval on {len(labeled_queries)} labeled queries "
        f"({n_answerable} answerable, {n_excluded} corpus-excluded, {n_off_topic} off-topic)..."
    )

    scored: list[tuple[LabeledQuery, float | None, list]] = []
    for i, labeled in enumerate(labeled_queries):
        result = retriever.retrieve(labeled.query)
        top1 = result.results[0].fused_score if result.results else None
        scored.append((labeled, top1, result.results))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(labeled_queries)} scored", flush=True)

    tau_abs_grid = [0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    tau_margin_grid = [0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02]

    sweep = [
        evaluate_threshold(scored, tau_abs, tau_margin)
        for tau_abs in tau_abs_grid
        for tau_margin in tau_margin_grid
    ]
    best = max(sweep, key=lambda r: r["abstain_f1"])

    output = {
        "strategy": args.strategy,
        "sample_size": len(labeled_queries),
        "sweep": sweep,
        "recommended": best,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Best operating point (by abstain_f1) ===")
    print(f"  tau_abs={best['tau_abs']}  tau_margin={best['tau_margin']}")
    print(
        f"  answer:   precision={best['answer_precision']:.3f} "
        f"recall={best['answer_recall']:.3f} f1={best['answer_f1']:.3f}"
    )
    print(
        f"  abstain:  precision={best['abstain_precision']:.3f} "
        f"recall={best['abstain_recall']:.3f} f1={best['abstain_f1']:.3f}"
    )
    print(f"  confusion: tp={best['tp']} fp={best['fp']} fn={best['fn']} tn={best['tn']}")
    print(f"\nFull sweep written to {args.output}")
    print("Set guardrails.tau_abs / guardrails.tau_margin in config/default.yaml to these values.")


if __name__ == "__main__":
    main()
