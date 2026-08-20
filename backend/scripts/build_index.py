from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import argparse
import json
import time
from pathlib import Path

from app.chunking.registry import STRATEGY_NAMES, get_strategy, requires_encoder
from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.models.chunk import Chunk, Document
from config.loader import load_config

# One row order shared by the dense index, the sparse index and the metadata store for
# a given strategy — see the module docstrings in app/indexing/{dense_index,sparse_index,
# store}.py. Building all three from the same `chunks` list in the same pass is what
# keeps that invariant true; don't split this into separate passes over `chunks`.

DONE_MARKER = "BUILD_COMPLETE"
# A batch pads to its longest member's token length, and self-attention memory scales
# with batch_size * seq_len^2. Most MS-MARCO passages are short (median ~55 words, see
# plan notes) but the tail runs long (observed max ~2285 words -> truncates to
# MAX_SEQ_LEN=512 tokens) — a batch that happens to contain one such outlier pads
# everything in it to 512, and at batch=256 that made ONNX Runtime request a single
# ~2.3GB allocation and crash on this machine's 16GB RAM. 32 keeps the worst case
# (batch=32, seq_len=512) around a few hundred MB, safely inside budget even under
# memory pressure from everything else running alongside this build.
ENCODE_BATCH = 32


def load_documents(corpus_path: Path) -> list[Document]:
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=row["passage_id"],
                    text=row["text"],
                    language=row["language"],
                    query_type=row.get("query_type"),
                )
            )
    return docs


def chunk_all(strategy_name: str, docs: list[Document], encoder: E5Encoder | None) -> list[Chunk]:
    strategy = get_strategy(strategy_name, encoder=encoder if requires_encoder(strategy_name) else None)
    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        chunks.extend(strategy.chunk(doc))
        if (i + 1) % 20000 == 0:
            print(f"    chunked {i + 1}/{len(docs)} docs -> {len(chunks)} chunks so far", flush=True)
    return chunks


def dense_vectors_for(strategy_name: str, chunks: list[Chunk], encoder: E5Encoder):
    """Returns an (n_chunks, dim) float32 array. late_chunking already computed its
    context-pooled vectors during chunking (that's the whole point of the strategy) —
    reuse those instead of re-encoding chunk text standalone, which would throw away
    the document context the strategy exists to capture."""
    import numpy as np

    if strategy_name == "late_chunking":
        return np.vstack([c.extra["context_vector"] for c in chunks]).astype("float32")

    vectors = []
    for start in range(0, len(chunks), ENCODE_BATCH):
        batch = chunks[start : start + ENCODE_BATCH]
        vectors.append(encoder.encode_passages([c.text for c in batch]))
        if (start // ENCODE_BATCH) % 20 == 0:
            print(f"    encoded {min(start + ENCODE_BATCH, len(chunks))}/{len(chunks)} chunks", flush=True)
    return np.vstack(vectors).astype("float32")


def build_strategy(
    strategy_name: str,
    docs: list[Document],
    encoder: E5Encoder,
    out_dir: Path,
    ef_construction: int,
    ef_search: int,
    force: bool,
) -> dict:
    strategy_dir = out_dir / strategy_name
    done_marker = strategy_dir / DONE_MARKER
    if done_marker.exists() and not force:
        print(f"[{strategy_name}] already built (found {DONE_MARKER}), skipping. Use --force to rebuild.")
        with open(done_marker, encoding="utf-8") as f:
            return json.load(f)

    print(f"[{strategy_name}] chunking {len(docs)} documents...", flush=True)
    t0 = time.perf_counter()
    chunks = chunk_all(strategy_name, docs, encoder)
    chunk_time_s = time.perf_counter() - t0
    print(f"[{strategy_name}] {len(chunks)} chunks from {len(docs)} docs ({chunk_time_s:.1f}s)", flush=True)

    print(f"[{strategy_name}] encoding dense vectors...", flush=True)
    t0 = time.perf_counter()
    vectors = dense_vectors_for(strategy_name, chunks, encoder)
    encode_time_s = time.perf_counter() - t0

    print(f"[{strategy_name}] building dense index (HNSW, efConstruction={ef_construction})...", flush=True)
    t0 = time.perf_counter()
    dense = DenseIndex(dim=vectors.shape[1], ef_construction=ef_construction, ef_search=ef_search)
    dense.add(vectors)
    dense_build_time_s = time.perf_counter() - t0

    print(f"[{strategy_name}] building sparse BM25 index...", flush=True)
    t0 = time.perf_counter()
    sparse = BM25SparseIndex()
    sparse.fit([c.text for c in chunks])
    sparse_build_time_s = time.perf_counter() - t0

    print(f"[{strategy_name}] writing metadata store...", flush=True)
    store = MetadataStore(strategy_dir / "meta.sqlite3")
    store.add_all(chunks)
    store.close()

    dense.save(strategy_dir / "dense.faiss")
    sparse.save(strategy_dir / "sparse")

    size_bytes = sum(p.stat().st_size for p in strategy_dir.rglob("*") if p.is_file())
    stats = {
        "strategy": strategy_name,
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "avg_chunks_per_doc": len(chunks) / len(docs) if docs else 0.0,
        "vector_dim": int(vectors.shape[1]),
        "chunk_time_s": chunk_time_s,
        "encode_time_s": encode_time_s,
        "dense_build_time_s": dense_build_time_s,
        "sparse_build_time_s": sparse_build_time_s,
        "total_build_time_s": chunk_time_s + encode_time_s + dense_build_time_s + sparse_build_time_s,
        "index_size_bytes": size_bytes,
        "index_size_mb": round(size_bytes / (1024 * 1024), 1),
    }
    with open(done_marker, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{strategy_name}] done: {stats['n_chunks']} chunks, {stats['index_size_mb']}MB, "
          f"{stats['total_build_time_s']:.1f}s total", flush=True)
    return stats


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--output-dir", default="data/index")
    parser.add_argument(
        "--strategies",
        default=",".join(config.chunking.strategies),
        help="comma-separated strategy names, default: config's chunking.strategies",
    )
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=config.retrieval.ef_search)
    parser.add_argument("--force", action="store_true", help="rebuild even if already done")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for name in strategies:
        if name not in STRATEGY_NAMES:
            raise SystemExit(f"unknown strategy {name!r} (known: {', '.join(STRATEGY_NAMES)})")

    print(f"Loading corpus from {args.corpus}...")
    docs = load_documents(Path(args.corpus))
    print(f"Loaded {len(docs)} documents.")

    print("Loading encoder (one-time cost, shared across all strategies)...")
    encoder = E5Encoder()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for name in strategies:
        all_stats[name] = build_strategy(
            name, docs, encoder, out_dir, args.ef_construction, args.ef_search, args.force
        )

    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print("\n=== Index build summary ===")
    for name, stats in all_stats.items():
        print(f"  {name}: {stats['n_chunks']} chunks, {stats['index_size_mb']}MB, "
              f"{stats['total_build_time_s']:.1f}s")


if __name__ == "__main__":
    main()
