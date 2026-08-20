from __future__ import annotations

from app.chunking.strategies import (
    FixedWindowChunker,
    LateChunkingChunker,
    MetadataAwareChunker,
    RecursiveChunker,
    SemanticBreakpointChunker,
)
from app.models.chunk import ChunkStrategy

# Strategies needing an encoder are constructed lazily so the cheap ones stay usable
# (and unit-testable) without loading a model.
_ENCODER_FREE = {
    "fixed_256_overlap_64": lambda: FixedWindowChunker(window=256, overlap=64),
    "recursive_512": lambda: RecursiveChunker(target=512, overlap=50),
    "metadata_aware": lambda: MetadataAwareChunker(target=512),
}
_ENCODER_BOUND = {
    "semantic_breakpoint": lambda enc: SemanticBreakpointChunker(enc, percentile=95),
    "late_chunking": lambda enc: LateChunkingChunker(enc, target=512, overlap=50),
}

STRATEGY_NAMES = list(_ENCODER_FREE) + list(_ENCODER_BOUND)


def requires_encoder(name: str) -> bool:
    return name in _ENCODER_BOUND


def get_strategy(name: str, encoder=None) -> ChunkStrategy:
    if name in _ENCODER_FREE:
        return _ENCODER_FREE[name]()
    if name in _ENCODER_BOUND:
        if encoder is None:
            raise ValueError(f"strategy {name!r} requires an encoder")
        return _ENCODER_BOUND[name](encoder)
    raise KeyError(f"unknown chunking strategy: {name!r} (known: {', '.join(STRATEGY_NAMES)})")
