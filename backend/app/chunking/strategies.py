from __future__ import annotations

import numpy as np

from app.chunking.base import make_chunk, split_sentences, word_spans
from app.models.chunk import Chunk, Document


class FixedWindowChunker:
    """Baseline: fixed token windows with overlap, ignoring all structure."""

    def __init__(self, window: int = 256, overlap: int = 64):
        if overlap >= window:
            raise ValueError("overlap must be smaller than window")
        self.window = window
        self.overlap = overlap
        self.name = f"fixed_{window}_overlap_{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = word_spans(doc.text)
        if not spans:
            return []
        step = self.window - self.overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(spans), step)):
            window = spans[start : start + self.window]
            if not window:
                break
            cs, ce = window[0][0], window[-1][1]
            chunks.append(make_chunk(doc, doc.text[cs:ce], cs, ce, self.name, i))
            if start + self.window >= len(spans):
                break
        return chunks


class RecursiveChunker:
    """Splits on paragraph, then sentence (Indic-aware), then word boundaries."""

    def __init__(self, target: int = 512, overlap: int = 50):
        self.target = target
        self.overlap = overlap
        self.name = f"recursive_{target}"

    def chunk(self, doc: Document) -> list[Chunk]:
        units = self._split_units(doc.text)
        if not units:
            return []

        chunks: list[Chunk] = []
        current: list[tuple[int, int]] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            cs, ce = current[0][0], current[-1][1]
            chunks.append(make_chunk(doc, doc.text[cs:ce], cs, ce, self.name, len(chunks)))
            if self.overlap > 0:
                kept: list[tuple[int, int]] = []
                kept_len = 0
                for span in reversed(current):
                    span_len = len(word_spans(doc.text[span[0] : span[1]]))
                    if kept_len + span_len > self.overlap:
                        break
                    kept.insert(0, span)
                    kept_len += span_len
                current, current_len = kept, kept_len
            else:
                current, current_len = [], 0

        for span in units:
            span_len = len(word_spans(doc.text[span[0] : span[1]]))
            if current and current_len + span_len > self.target:
                flush()
            current.append(span)
            current_len += span_len
        flush()
        return chunks

    def _split_units(self, text: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        for para_start, para_end in self._paragraph_spans(text):
            para = text[para_start:para_end]
            cursor = para_start
            for sentence in split_sentences(para):
                idx = text.find(sentence, cursor, para_end)
                if idx == -1:
                    continue
                units.append((idx, idx + len(sentence)))
                cursor = idx + len(sentence)
        return units

    def _paragraph_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for block in text.split("\n\n"):
            if block.strip():
                start = text.find(block, cursor)
                spans.append((start, start + len(block)))
                cursor = start + len(block)
        return spans or ([(0, len(text))] if text.strip() else [])


class SemanticBreakpointChunker:
    """Splits where consecutive-sentence embedding distance exceeds a percentile."""

    name = "semantic_breakpoint"

    def __init__(self, encoder, percentile: int = 95, max_sentences: int = 200):
        self.encoder = encoder
        self.percentile = percentile
        self.max_sentences = max_sentences

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)[: self.max_sentences]
        if len(sentences) < 2:
            return _whole_document(doc, self.name)

        vectors = self.encoder.encode_passages(sentences)
        # Vectors are L2-normalized, so the dot product is cosine similarity.
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))

        groups: list[list[str]] = [[sentences[0]]]
        for sentence, distance in zip(sentences[1:], distances):
            if distance >= threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        return _chunks_from_groups(doc, groups, self.name)


class LateChunkingChunker:
    """Encodes the whole passage once, then mean-pools token vectors per chunk.

    The original spec assumed BGE-M3's 8k context. e5-small caps at 512 tokens, so the
    pooled context is passage-wide rather than document-wide — a real reduction from the
    spec's intent, documented rather than papered over.
    """

    name = "late_chunking"

    def __init__(self, encoder, target: int = 512, overlap: int = 50):
        self.encoder = encoder
        self._boundaries = RecursiveChunker(target=target, overlap=overlap)

    def chunk(self, doc: Document) -> list[Chunk]:
        base = self._boundaries.chunk(doc)
        if not base:
            return []
        pooled = self.encoder.encode_passages_with_context([c.text for c in base], doc.text)
        return [
            make_chunk(
                doc,
                c.text,
                c.char_start,
                c.char_end,
                self.name,
                i,
                context_vector=pooled[i].tolist(),
            )
            for i, c in enumerate(base)
        ]


class MetadataAwareChunker:
    """Respects passage boundaries and carries a filterable payload for pre-filtered ANN."""

    name = "metadata_aware"

    def __init__(self, target: int = 512):
        self.target = target

    def chunk(self, doc: Document) -> list[Chunk]:
        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for sentence in split_sentences(doc.text):
            sentence_len = len(sentence.split())
            if current and current_len + sentence_len > self.target:
                groups.append(current)
                current, current_len = [], 0
            current.append(sentence)
            current_len += sentence_len
        if current:
            groups.append(current)
        if not groups:
            return _whole_document(doc, self.name)

        chunks = _chunks_from_groups(doc, groups, self.name)
        for chunk in chunks:
            chunk.extra.update(
                {
                    "filter_language": doc.language,
                    "filter_query_type": doc.query_type,
                    "filter_doc_id": doc.doc_id,
                }
            )
        return chunks


def _chunks_from_groups(doc: Document, groups: list[list[str]], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    for i, group in enumerate(groups):
        joined = " ".join(group)
        start = doc.text.find(group[0], cursor)
        if start == -1:
            start = cursor
        end = min(start + len(joined), len(doc.text))
        chunks.append(make_chunk(doc, doc.text[start:end], start, end, strategy, i))
        cursor = end
    return chunks


def _whole_document(doc: Document, strategy: str) -> list[Chunk]:
    if not doc.text.strip():
        return []
    return [make_chunk(doc, doc.text, 0, len(doc.text), strategy, 0)]
