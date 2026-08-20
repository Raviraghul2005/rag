"""Loads backend/.env and sets sane cache-location defaults, before anything else in
the process touches os.environ or the Hugging Face cache path. Import this module
first — as the very first import — in every process entry point (app.main, scripts/*).

Two real bugs this exists to prevent:
1. python-dotenv was a declared dependency that nothing actually called — GROQ_API_KEY
   etc. lived only in backend/.env, so any code reading os.environ["GROQ_API_KEY"]
   directly (app/generation/providers.py) would KeyError at first real request.
2. huggingface_hub's default model-download cache (HF_HOME) sits on C:, which was down
   to ~14GB free during this build. app/data/languages.py and app/indexing/embeddings.py
   already redirect *their own* downloads to D: via custom env vars, but any plain
   `AutoModel.from_pretrained(...)` call elsewhere (e.g. app/guardrails/grounding.py)
   still falls through to the C: default unless HF_HOME itself is redirected here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolved relative to this file, not the process's cwd: dotenv's own auto-search
# (find_dotenv() with no path) proved cwd-dependent in testing — it silently found
# nothing when a script was launched from the repo root instead of backend/, which
# would have shipped a backend that only loads its secrets if launched from the right
# directory. Deployment (Docker CMD, uvicorn, a test runner) shouldn't have to get that
# right. In production (spec §16.2) there is no .env file at all — secrets arrive via
# Space Secrets as real environment variables — so load_dotenv() finding nothing here
# is the expected, harmless case, not an error.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("HF_HOME", r"D:\dev-cache\raingoa\hf_home")
