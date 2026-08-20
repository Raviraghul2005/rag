from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Spec's original design used BGE-M3's own lexical weights for the sparse side (one
# model, one forward pass, both dense and sparse vectors). The CPU pivot to e5-small
# (see app/indexing/embeddings.py) drops that — e5-small yields dense vectors only.
# BM25 over a classic term-document matrix is the documented substitute: still an
# inverted index doing exact term / named-entity matching, still complements dense
# retrieval the same way, just built from raw token counts instead of learned weights.
#
# Not `re.findall(r"\w+", ...)`: Python's \w excludes combining marks (Unicode category
# Mn/Mc), which silently shatters Indic words at every dependent vowel sign — भारत
# splits into "भ" + "ारत" because ा (U+093E) fails \w. Classifying by Unicode category
# (Letter/Mark/Number run together) keeps the base letter and its matras as one token,
# uniformly across every script here, with no per-language hardcoding.
_TOKEN_CATEGORIES = ("L", "M", "N")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in _TOKEN_CATEGORIES:
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


class BM25SparseIndex:
    """Okapi BM25 over a scipy sparse term-document matrix.

    Built once per chunking strategy from that strategy's chunk texts, in the same row
    order as the strategy's dense index and metadata store (chunk_id_order is the shared
    key across all three — see scripts/build_index.py).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.doc_term_counts: sp.csr_matrix | None = None  # (n_docs, n_vocab), raw counts
        self.doc_lengths: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.idf: np.ndarray | None = None  # (n_vocab,)

    def fit(self, texts: list[str]) -> None:
        tokenized = [tokenize(t) for t in texts]

        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab

        n_docs, n_vocab = len(texts), len(vocab)
        rows, cols, data = [], [], []
        doc_lengths = np.zeros(n_docs, dtype=np.float64)
        df = np.zeros(n_vocab, dtype=np.float64)
        for row, tokens in enumerate(tokenized):
            doc_lengths[row] = len(tokens)
            counts: dict[int, int] = {}
            for tok in tokens:
                col = vocab[tok]
                counts[col] = counts.get(col, 0) + 1
            for col, count in counts.items():
                rows.append(row)
                cols.append(col)
                data.append(count)
                df[col] += 1

        self.doc_term_counts = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_docs, n_vocab), dtype=np.float64
        )
        self.doc_lengths = doc_lengths
        self.avgdl = float(doc_lengths.mean()) if n_docs else 0.0
        # Robertson-Sparck Jones idf, floored at a small positive value so unseen-heavy
        # terms (df close to n_docs) don't push scores negative.
        self.idf = np.maximum(
            np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0), 1e-9
        )

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns (row_index, bm25_score) pairs, sorted descending, for query terms
        that exist in the vocabulary. Rows are index positions into the fitted corpus,
        same order as the strategy's dense index and metadata store."""
        if self.doc_term_counts is None:
            raise RuntimeError("index not fit or loaded")

        query_cols = [self.vocab[tok] for tok in tokenize(query) if tok in self.vocab]
        if not query_cols:
            return []

        n_docs = self.doc_term_counts.shape[0]
        scores = np.zeros(n_docs, dtype=np.float64)
        norm = 1.0 - self.b + self.b * (self.doc_lengths / self.avgdl if self.avgdl else 0.0)
        for col in set(query_cols):
            tf = np.asarray(self.doc_term_counts[:, col].todense()).ravel()
            hit = tf > 0
            if not hit.any():
                continue
            scores[hit] += (
                self.idf[col] * (tf[hit] * (self.k1 + 1)) / (tf[hit] + self.k1 * norm[hit])
            )

        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        order = nonzero[np.argsort(-scores[nonzero])][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(dir_path / "doc_term_counts.npz", self.doc_term_counts)
        np.save(dir_path / "doc_lengths.npy", self.doc_lengths)
        np.save(dir_path / "idf.npy", self.idf)
        with open(dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "avgdl": self.avgdl, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, dir_path: Path) -> "BM25SparseIndex":
        with open(dir_path / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        index = cls(k1=meta["k1"], b=meta["b"])
        index.vocab = meta["vocab"]
        index.avgdl = meta["avgdl"]
        index.doc_term_counts = sp.load_npz(dir_path / "doc_term_counts.npz").tocsr()
        index.doc_lengths = np.load(dir_path / "doc_lengths.npy")
        index.idf = np.load(dir_path / "idf.npy")
        return index
