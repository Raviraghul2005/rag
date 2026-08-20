from __future__ import annotations

import os

from openai import AsyncOpenAI

# Both Groq and Cerebras publish OpenAI-compatible chat-completions endpoints (verified
# against their live APIs, not assumed — see config/default.yaml's generation block for
# the model-catalog note: llama-3.1-8b-instant was retired from Groq before this build
# reached Phase 6). Model names live in config, not here (spec §21) — this module only
# owns the base URLs and how each client authenticates.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)


def cerebras_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["CEREBRAS_API_KEY"], base_url=CEREBRAS_BASE_URL)
