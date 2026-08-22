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
# everything in it to 512.
#
# CPU path: at batch=256 that made ONNX Runtime request a single ~2.3GB allocation and
# crash under this machine's shared 16GB system RAM (competing with everything else
# running). 32 keeps the worst case safely inside budget even under memory pressure.
#
# GPU path: fp16 on a dedicated 6GB VRAM pool (RTX 4050, not shared with the OS/other
# apps the way system RAM is) has real headroom for a much larger batch. Benchmarked
# live on this GPU post-warm-up: 32->1168/s, 128->3391/s, 256->4062/s passages/sec —
# 256 is the batch size actually used, ~9x the ideal CPU calibration's 438/s.
ENCODE_BATCH_CPU = 32
ENCODE_BATCH_GPU = 256


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


# Benchmarked live on this GPU, length-sorted (see below), post-warm-up: 16->391.7/s,
# 32->345.8/s, 64->391.4/s, 96->372.0/s, 128->400.7/s docs/sec. 128 both won outright
# and matches ENCODE_BATCH_GPU's established value, so no new constant to reason about.
CHUNK_BATCH_GPU = 128


def chunk_all(strategy_name: str, docs: list[Document], encoder: E5Encoder | None) -> list[Chunk]:
    strategy = get_strategy(strategy_name, encoder=encoder if requires_encoder(strategy_name) else None)
    chunks: list[Chunk] = []
    if hasattr(strategy, "chunk_batch") and encoder is not None and encoder.device == "cuda":
        batch_size = CHUNK_BATCH_GPU
        # Padding pads every sequence in a batch to that batch's longest member, and
        # attention cost scales with seq_len^2 -- an unsorted batch of docs with wildly
        # different lengths wastes real compute on padding. Sorting by length first so
        # each batch is length-homogeneous measured 2.3x faster than unsorted batching
        # at the same batch size (332 vs 162 docs/sec at batch=256) -- final chunk order
        # doesn't matter, every Chunk carries its own doc_id/chunk_index for identity.
        order = sorted(range(len(docs)), key=lambda i: len(docs[i].text))
        sorted_docs = [docs[i] for i in order]
        for start in range(0, len(sorted_docs), batch_size):
            batch = sorted_docs[start : start + batch_size]
            chunks.extend(strategy.chunk_batch(batch))
            done = min(start + batch_size, len(sorted_docs))
            if (start // batch_size) % 10 == 0 or done == len(sorted_docs):
                print(f"    chunked {done}/{len(sorted_docs)} docs -> {len(chunks)} chunks so far", flush=True)
        return chunks

    for i, doc in enumerate(docs):
        chunks.extend(strategy.chunk(doc))
        if (i + 1) % 20000 == 0:
            print(f"    chunked {i + 1}/{len(docs)} docs -> {len(chunks)} chunks so far", flush=True)
    return chunks


def dense_vectors_for(strategy_name: str, chunks: list[Chunk], encoder: E5Encoder):
    """Returns an (n_chunks, dim) float32 array. late_chunking already computed its
    context-pooled vectors during chunking (that's the whole point of the strategy) —
    reuse those instead of re-encoding chunk text standalone, which would throw away
    the document context the strategy exists to capture.

    Pre-allocated and filled in place, freeing each chunk's stored copy as it's copied
    in — at large corpus sizes (840k+ chunks), building a full Python list of every
    chunk's vector via list-comprehension + np.vstack held two full copies of the data
    in memory at once (the per-chunk copies still on `chunks`, plus vstack's own
    concatenated output) and was enough to exhaust this machine's RAM outright.
    """
    import numpy as np

    if strategy_name == "late_chunking":
        dim = len(chunks[0].extra["context_vector"])
        vectors = np.empty((len(chunks), dim), dtype="float32")
        for i, c in enumerate(chunks):
            vectors[i] = c.extra["context_vector"]
            c.extra["context_vector"] = None
        return vectors

    batch_size = ENCODE_BATCH_GPU if encoder.device == "cuda" else ENCODE_BATCH_CPU
    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.append(encoder.encode_passages([c.text for c in batch]))
        if (start // batch_size) % 20 == 0:
            print(f"    encoded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks", flush=True)
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
