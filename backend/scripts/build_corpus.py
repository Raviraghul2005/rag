from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from app.data.languages import ALL_LANGUAGES, HF_CACHE_DIR, VALIDATION_FILE, script_fraction
from app.models.corpus import CorpusPassage, EvalQuery

# Only what's needed — skipping English_passages/Eng_* /meta/Answer cuts the read
# substantially, which matters because each file is a single ~1.2GB row group.
COLUMNS = ["query", "query_id", "query_type", "passages"]


def local_parquet(lang: str) -> str:
    return hf_hub_download(
        "ai4bharat/MSMARCO-XI",
        f"validation/{VALIDATION_FILE[lang]}",
        repo_type="dataset",
        cache_dir=str(HF_CACHE_DIR),
    )


def process_language(
    lang: str,
    n_passages: int,
    seed: int,
    held_out_frac: float,
    min_script_fraction: float,
) -> tuple[list[CorpusPassage], list[EvalQuery], dict]:
    rng = random.Random(f"{seed}:{lang}")
    dedup_index: dict[str, str] = {}
    corpus: list[CorpusPassage] = []
    eval_queries: list[EvalQuery] = []
    stats: Counter[str] = Counter()

    # Sampling is query-driven but budgeted by passage count: whole queries are always
    # consumed intact so `is_selected` relevance labels never get orphaned (plan §5).
    pf = pq.ParquetFile(local_parquet(lang))
    for batch in pf.iter_batches(batch_size=512, columns=COLUMNS):
        if len(corpus) >= n_passages:
            break
        for row in batch.to_pylist():
            if len(corpus) >= n_passages:
                break
            stats["rows_seen"] += 1
            query_id = str(row["query_id"])
            query_type = row.get("query_type")
            passages = row["passages"] or {}
            translated = passages.get("Translated_passages") or []
            selected_flags = passages.get("is_selected") or []

            held_out = rng.random() < held_out_frac
            relevant_ids: list[str] = []
            kept_any = False

            for idx, text in enumerate(translated):
                stats["passages_seen"] += 1
                selected = bool(selected_flags[idx]) if idx < len(selected_flags) else False

                if not text or not text.strip():
                    stats["excluded_empty"] += 1
                    continue
                if script_fraction(text, lang) < min_script_fraction:
                    stats["excluded_script_mismatch"] += 1
                    continue

                passage_id = f"{lang}_{query_id}_{idx}"

                if held_out and selected:
                    # The passage that answers this query is deliberately withheld from
                    # the corpus, making the query genuinely unanswerable — the
                    # "corpus-excluded" bucket the abstention curve needs (spec §11.4).
                    # Its distractor passages still enter the corpus below.
                    relevant_ids.append(passage_id)
                    stats["held_out_passages_skipped"] += 1
                    continue

                normalized = " ".join(text.split()).lower()
                if normalized in dedup_index:
                    passage_id = dedup_index[normalized]
                    stats["deduped"] += 1
                else:
                    dedup_index[normalized] = passage_id
                    corpus.append(
                        CorpusPassage(
                            passage_id=passage_id,
                            language=lang,
                            query_type=query_type,
                            is_selected=selected,
                            source_query_id=query_id,
                            text=text,
                        )
                    )
                kept_any = True
                if selected:
                    relevant_ids.append(passage_id)

            # A held-out query with no surviving distractors would vanish from the
            # corpus entirely, making it indistinguishable from an off-topic query.
            if relevant_ids and (kept_any or not held_out):
                eval_queries.append(
                    EvalQuery(
                        query_id=query_id,
                        query=row["query"],
                        language=lang,
                        query_type=query_type,
                        relevant_passage_ids=relevant_ids,
                        held_out=held_out,
                    )
                )

    return corpus, eval_queries, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=",".join(ALL_LANGUAGES))
    parser.add_argument(
        "--n-per-language",
        type=int,
        required=True,
        help="target corpus passages per language (not queries) - default for languages not named in --overrides",
    )
    parser.add_argument(
        "--overrides",
        default="",
        help="per-language target overrides, e.g. 'hi=150000,ta=150000' - takes precedence over --n-per-language for named languages",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-frac", type=float, default=0.15)
    parser.add_argument("--min-script-fraction", type=float, default=0.3)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, int] = {}
    for pair in args.overrides.split(","):
        pair = pair.strip()
        if not pair:
            continue
        lang, _, n = pair.partition("=")
        overrides[lang.strip()] = int(n.strip())

    all_corpus: list[CorpusPassage] = []
    all_eval: list[EvalQuery] = []
    per_language: dict[str, dict] = {}

    for lang in languages:
        n_target = overrides.get(lang, args.n_per_language)
        print(f"[{lang}] reading (target {n_target})...", flush=True)
        corpus, eval_queries, stats = process_language(
            lang,
            n_passages=n_target,
            seed=args.seed,
            held_out_frac=args.held_out_frac,
            min_script_fraction=args.min_script_fraction,
        )
        all_corpus.extend(corpus)
        all_eval.extend(eval_queries)
        per_language[lang] = {
            "corpus_passages": len(corpus),
            "eval_queries": len(eval_queries),
            "held_out_queries": sum(1 for q in eval_queries if q.held_out),
            **stats,
        }
        print(f"  -> {len(corpus)} passages, {len(eval_queries)} eval queries", flush=True)

    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for p in all_corpus:
            f.write(p.model_dump_json() + "\n")
    with open(out_dir / "eval_set.jsonl", "w", encoding="utf-8") as f:
        for q in all_eval:
            f.write(q.model_dump_json() + "\n")
    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(per_language, f, indent=2, ensure_ascii=False)

    print("\n=== Corpus statistics ===")
    for lang, s in per_language.items():
        print(f"  {lang}: {s}")
    print(f"\nTotal passages: {len(all_corpus)}")
    print(f"Total eval queries: {len(all_eval)} (need >= 500)")
    print(f"Held-out queries: {sum(1 for q in all_eval if q.held_out)}")


if __name__ == "__main__":
    main()
