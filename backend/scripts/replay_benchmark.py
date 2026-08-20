from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from app.generation.generator import Generator
from app.guardrails.grounding import GroundingVerifier
from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.models.corpus import EvalQuery
from app.models.pipeline import PipelineRequest
from app.pipeline import run_pipeline
from app.retrieval.retriever import Retriever
from config.loader import GuardrailsConfig, load_config

# The deterministic replay runner spec §12.8 and §13.3 ask for: a fixed query set that
# regenerates the entire latency table by running one documented command, so a judge
# can rerun the numbers rather than trust a claimed screenshot. Cold and warm runs are
# reported separately per §13.3 — the first request pays model/index warm-up cost even
# though main.py's own startup already warms the encoder+index, because this script is
# its own process and hasn't paid that cost yet.


def load_stratified_sample(eval_set_path: Path, sample_size: int, seed: int) -> list[EvalQuery]:
    by_language: dict[str, list[EvalQuery]] = {}
    with open(eval_set_path, encoding="utf-8") as f:
        for line in f:
            eq = EvalQuery(**json.loads(line))
            by_language.setdefault(eq.language, []).append(eq)

    rng = random.Random(seed)
    for queries in by_language.values():
        rng.shuffle(queries)

    # Round-robin across languages so the sample stays stratified even when
    # sample_size doesn't divide evenly across however many languages have data.
    languages = sorted(by_language)
    sample: list[EvalQuery] = []
    cursors = dict.fromkeys(languages, 0)
    while len(sample) < sample_size and any(cursors[lang] < len(by_language[lang]) for lang in languages):
        for lang in languages:
            if cursors[lang] < len(by_language[lang]):
                sample.append(by_language[lang][cursors[lang]])
                cursors[lang] += 1
                if len(sample) >= sample_size:
                    break
    return sample


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def summarize(label: str, timings_per_query: list[dict[str, float]]) -> dict:
    stage_names = sorted({name for t in timings_per_query for name in t})
    totals = [sum(t.values()) for t in timings_per_query]
    per_stage = {
        stage: {
            "p50": percentile([t.get(stage, 0.0) for t in timings_per_query], 50),
            "p70": percentile([t.get(stage, 0.0) for t in timings_per_query], 70),
            "p100": percentile([t.get(stage, 0.0) for t in timings_per_query], 100),
        }
        for stage in stage_names
    }
    return {
        "label": label,
        "n": len(timings_per_query),
        "total_ms": {"p50": percentile(totals, 50), "p70": percentile(totals, 70), "p100": percentile(totals, 100)},
        "per_stage_ms": per_stage,
    }


async def run_sample(
    queries: list[EvalQuery], retriever: Retriever, generator: Generator,
    grounding_verifier: GroundingVerifier | None, guardrails: GuardrailsConfig,
) -> list[dict]:
    timings = []
    outcomes: dict[str, int] = {}
    for i, eq in enumerate(queries):
        response = await run_pipeline(
            PipelineRequest(transcript=eq.query, language=eq.language),
            retriever, generator, grounding_verifier, guardrails,
        )
        timings.append(response.timings.stages)
        outcomes[response.outcome] = outcomes.get(response.outcome, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(queries)} done", flush=True)
    print(f"    outcomes: {outcomes}")
    return timings


def load_calibrated_guardrails(config, calibration_path: Path) -> GuardrailsConfig:
    if not calibration_path.exists():
        print(f"  no calibration file at {calibration_path}, using config defaults (gates likely skipped)")
        return config.guardrails
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    recommended = calibration["recommended"]
    return config.guardrails.model_copy(
        update={"tau_abs": recommended["tau_abs"], "tau_margin": recommended["tau_margin"]}
    )


async def main_async() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_set.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--strategy", default=config.chunking.active_strategy)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-file", default="data/calibration_results.json")
    parser.add_argument("--no-grounding", action="store_true", help="report the pipeline without grounding (spec 11.3's cost-of-safety comparison)")
    parser.add_argument("--output", default="data/latency_benchmark.json")
    args = parser.parse_args()

    strategy_dir = Path(args.index_dir) / args.strategy
    print(f"Loading strategy={args.strategy}...")
    encoder = E5Encoder()
    retriever = Retriever(
        encoder,
        DenseIndex.load(strategy_dir / "dense.faiss", ef_search=config.retrieval.ef_search),
        BM25SparseIndex.load(strategy_dir / "sparse"),
        MetadataStore(strategy_dir / "meta.sqlite3"),
        config.retrieval,
    )
    generator = Generator(config.generation, config.harness)
    grounding_verifier = None if args.no_grounding else GroundingVerifier()
    guardrails = load_calibrated_guardrails(config, Path(args.calibration_file))
    if args.no_grounding:
        guardrails = guardrails.model_copy(update={"enable_grounding": False})

    queries = load_stratified_sample(Path(args.eval_set), args.sample_size, args.seed)
    print(f"Loaded {len(queries)} stratified queries across {len({q.language for q in queries})} languages.")

    print("Cold run (1 query, process just started)...")
    cold_response = await run_pipeline(
        PipelineRequest(transcript=queries[0].query, language=queries[0].language),
        retriever, generator, grounding_verifier, guardrails,
    )
    cold_summary = summarize("cold", [cold_response.timings.stages])

    warm_queries = queries[1:]
    print(f"Warm run ({len(warm_queries)} queries)...")
    start = time.perf_counter()
    warm_timings = await run_sample(warm_queries, retriever, generator, grounding_verifier, guardrails)
    elapsed_s = time.perf_counter() - start
    warm_summary = summarize("warm", warm_timings)

    output = {
        "strategy": args.strategy,
        "grounding_enabled": not args.no_grounding,
        "sample_size": len(queries),
        "wall_clock_s": elapsed_s,
        "cold": cold_summary,
        "warm": warm_summary,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Warm run: total (ms) ===")
    print(f"  P50={warm_summary['total_ms']['p50']:.1f}  P70={warm_summary['total_ms']['p70']:.1f}  P100={warm_summary['total_ms']['p100']:.1f}")
    print("=== Warm run: per-stage P50 (ms) ===")
    for stage, values in warm_summary["per_stage_ms"].items():
        print(f"  {stage}: {values['p50']:.1f}")
    print(f"\nWritten to {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
