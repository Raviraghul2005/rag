from __future__ import annotations

import re

from app.models.chunk import Chunk, Document

# Sentence terminators across the corpus's 14 languages. The Devanagari danda (U+0964)
# and double danda (U+0965) are the ones a Latin-only splitter misses entirely — they
# terminate sentences in Hindi, Marathi, Sanskrit, Nepali, Bengali, Assamese and more.
# Urdu uses its own full stop (U+06D4) and question mark (U+061F).
SENTENCE_TERMINATORS = "।॥.?!۔؟"
_SENTENCE_SPLIT_RE = re.compile(rf"(?<=[{re.escape(SENTENCE_TERMINATORS)}])\s+")

# Whitespace tokenization throughout. Indic scripts don't segment on whitespace the way
# the model's subword tokenizer does, so "tokens" here are approximate — used for
# consistent, cheap chunk sizing, not for exact model-token accounting.
_WORD_RE = re.compile(r"\S+")


def split_sentences(text: str) -> list[str]:
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT_RE.split(text)) if s]


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def make_chunk(
    doc: Document, text: str, char_start: int, char_end: int, strategy: str, index: int, **extra
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{strategy}::{index}",
        text=text,
        doc_id=doc.doc_id,
        language=doc.language,
        query_type=doc.query_type,
        char_start=char_start,
        char_end=char_end,
        strategy=strategy,
        extra=extra,
    )
