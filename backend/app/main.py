from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.generation.generator import Generator
from app.guardrails.grounding import GroundingVerifier
from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.logging_setup import configure_logging
from app.models.pipeline import PipelineRequest, PipelineResponse
from app.pipeline import run_pipeline
from app.retrieval.retriever import Retriever
from app.stt.voice_session import run_voice_session
from config.loader import load_config

configure_logging()
config = load_config()
logger = logging.getLogger(__name__)

# Populated by lifespan at startup; a plain dict (not globals) so tests can substitute
# fakes into it directly without needing the real index/models on disk. Phase 11 wires
# this into a real /health readiness report — deliberately not done here yet, so the
# existing Phase 1 liveness check (a bare "server process is up") keeps meaning that
# and doesn't start requiring a multi-GB model+index load just to answer a ping.
resources: dict = {"ready": False}


def _resolve_strategy(requested: str | None) -> str:
    strategy = requested or config.chunking.active_strategy
    if strategy not in resources.get("retrievers", {}):
        raise HTTPException(status_code=400, detail=f"unknown or unbuilt strategy: {strategy!r}")
    return strategy


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        encoder = E5Encoder()
        retrievers: dict[str, Retriever] = {}
        # All strategies with a built index load at startup, not just the configured
        # default — the frontend's strategy selector (spec §15 item 5) needs to switch
        # retrieval live, which only proves anything if every strategy is a real,
        # independently loaded index rather than a label on one shared retriever.
        # A strategy whose index hasn't been built yet (data/index/<name>/ missing)
        # is skipped, not fatal — lets the server come up with whatever's ready.
        for strategy_name in config.chunking.strategies:
            strategy_dir = Path("data/index") / strategy_name
            if not (strategy_dir / "BUILD_COMPLETE").exists():
                logger.warning("strategy %s has no built index, skipping", strategy_name)
                continue
            retrievers[strategy_name] = Retriever(
                encoder,
                DenseIndex.load(strategy_dir / "dense.faiss", ef_search=config.retrieval.ef_search),
                BM25SparseIndex.load(strategy_dir / "sparse"),
                MetadataStore(strategy_dir / "meta.sqlite3"),
                config.retrieval,
            )

        if config.chunking.active_strategy not in retrievers:
            raise RuntimeError(
                f"configured active_strategy {config.chunking.active_strategy!r} has no built index"
            )

        # Warm the encoder + every loaded index before serving real traffic (spec
        # §16.2) — first inference on a freshly loaded ONNX/FAISS index pays a
        # one-time graph/cache cost that shouldn't land on the first real user's
        # request, for whichever strategy they happen to pick.
        for retriever in retrievers.values():
            retriever.retrieve("भारत")

        resources["retrievers"] = retrievers
        resources["generator"] = Generator(config.generation, config.harness)
        resources["grounding_verifier"] = (
            GroundingVerifier() if config.guardrails.enable_grounding else None
        )
        resources["ready"] = True
        logger.info("pipeline resources loaded, strategies=%s", list(retrievers))
    except Exception:
        logger.exception("failed to load pipeline resources at startup")
        resources["ready"] = False

    yield
    resources.clear()
    resources["ready"] = False


app = FastAPI(title="RAGInGoa", lifespan=lifespan)

# Never "*" — the key-leak concern in spec §4 depends on this staying an explicit allowlist.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://frontend-lilac-two-49.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    # Real model/index/provider state wires in during deployment (Phase 11)
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> dict[str, object]:
    """Lets the frontend's strategy selector populate itself from what's actually
    loaded, rather than hardcoding the list of five and risking drift."""
    return {
        "available": list(resources.get("retrievers", {})),
        "default": config.chunking.active_strategy,
    }


@app.post("/query", response_model=PipelineResponse)
async def query(request: PipelineRequest) -> PipelineResponse:
    if not resources.get("ready"):
        raise HTTPException(status_code=503, detail="pipeline resources not loaded yet")
    strategy = _resolve_strategy(request.strategy)
    return await run_pipeline(
        request,
        resources["retrievers"][strategy],
        resources["generator"],
        resources.get("grounding_verifier"),
        config.guardrails,
        strategy_used=strategy,
    )


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket, language: str = "auto", strategy: str | None = None) -> None:
    """Voice entry point (spec §15): the browser streams raw PCM16 audio here, never
    to Sarvam directly. `language` is a Sarvam BCP-47 code (e.g. "hi-IN") or "auto" —
    passed straight through to Sarvam, no translation needed here since the frontend
    is the one presenting language choices to the user in the first place. `strategy`
    is resolved once per connection (a voice session is one topic/conversation, unlike
    /query's per-request selector).
    """
    if not resources.get("ready"):
        await websocket.close(code=1013, reason="pipeline resources not loaded yet")
        return

    try:
        resolved_strategy = _resolve_strategy(strategy)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail))
        return

    await websocket.accept()
    try:
        await run_voice_session(
            websocket,
            resources["retrievers"][resolved_strategy],
            resources["generator"],
            resources.get("grounding_verifier"),
            config.guardrails,
            language_code=language,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("voice session crashed")
