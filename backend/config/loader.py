from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


class CorpusConfig(BaseModel):
    languages: list[str]
    passages_per_language: int | None
    seed: int


class ChunkingConfig(BaseModel):
    active_strategy: str
    strategies: list[str]


class RetrievalConfig(BaseModel):
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    rerank_candidates: int
    final_top_k: int
    ef_search: int


class GuardrailsConfig(BaseModel):
    tau_abs: float | None
    tau_margin: float | None
    grounding_threshold: float | None
    enable_input: bool
    enable_retrieval_gate: bool
    enable_grounding: bool


class GenerationConfig(BaseModel):
    primary: str
    primary_model: str
    primary_reasoning_effort: str | None = None
    failover: str
    failover_model: str
    max_tokens: int  # reported/enforced answer-length cap (spec §10's lever)
    max_tokens_request: int  # raw API token budget; see generator.py's reasoning-model note
    context_token_budget: int


class StageTimeoutMs(BaseModel):
    encode: int
    retrieve: int
    rerank: int
    generate: int


class HarnessConfig(BaseModel):
    max_retries: int
    stage_timeout_ms: StageTimeoutMs
    total_deadline_ms: int
    breaker_failure_threshold: int
    breaker_cooldown_s: int


class AppConfig(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    guardrails: GuardrailsConfig
    generation: GenerationConfig
    harness: HarnessConfig


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
