from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging_setup import configure_logging
from config.loader import load_config

configure_logging()
config = load_config()

app = FastAPI(title="RAGInGoa")

# Vercel prod domain gets added once deployed (Phase 11) — never "*", the key-leak
# concern in spec §4 depends on this staying an explicit allowlist.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
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
