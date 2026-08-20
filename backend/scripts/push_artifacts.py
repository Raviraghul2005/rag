from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse

from huggingface_hub import HfApi

from app.artifacts import DEFAULT_ARTIFACTS_REPO

# Uploads exactly what app/artifacts.py's ensure_artifacts() expects to find:
# corpus.jsonl, eval_set.jsonl, corpus_stats.json, and index/<strategy>/ per built
# strategy. Run this after scripts/build_corpus.py and scripts/build_index.py, before
# any fresh deploy that needs to pull them.
UPLOAD_PATTERNS = ["corpus.jsonl", "eval_set.jsonl", "corpus_stats.json", "index/**"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_ARTIFACTS_REPO)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--create", action="store_true", help="create the dataset repo if missing")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    if args.create:
        print(f"Ensuring dataset repo {args.repo_id} exists (private={args.private})...")
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    print(f"Uploading {args.data_dir}/ to {args.repo_id} (patterns: {UPLOAD_PATTERNS})...")
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=UPLOAD_PATTERNS,
    )
    print("Done.")


if __name__ == "__main__":
    main()
