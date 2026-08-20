import pytest

from app.chunking.base import split_sentences
from app.chunking.strategies import (
    FixedWindowChunker,
    MetadataAwareChunker,
    RecursiveChunker,
)
from app.models.chunk import Document

HINDI = "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। दिल्ली राजधानी है।"
URDU = "پاکستان ایک ملک ہے۔ یہاں اردو بولی جاتی ہے۔"
SANSKRIT = "अयं ग्रन्थः अस्ति॥ सः पठति॥"


def test_devanagari_danda_splits_sentences():
    assert len(split_sentences(HINDI)) == 3


def test_double_danda_splits_sentences():
    assert len(split_sentences(SANSKRIT)) == 2


def test_urdu_full_stop_splits_sentences():
    assert len(split_sentences(URDU)) == 2


def test_latin_period_still_splits():
    assert len(split_sentences("One sentence. Two sentences. Three.")) == 3


def test_danda_is_not_treated_as_plain_text():
    # The failure this guards against: a Latin-only splitter returns 1 for Hindi text,
    # silently producing one giant chunk per document. This is the detail spec §6 flags.
    assert len(split_sentences(HINDI)) > 1


@pytest.mark.parametrize("text", [HINDI, URDU, SANSKRIT])
@pytest.mark.parametrize(
    "chunker",
    [FixedWindowChunker(window=8, overlap=2), RecursiveChunker(target=8, overlap=2), MetadataAwareChunker(target=8)],
    ids=["fixed", "recursive", "metadata"],
)
def test_offsets_map_back_to_source_text(chunker, text):
    doc = Document(doc_id="d1", text=text, language="hi", query_type="DESCRIPTION")
    chunks = chunker.chunk(doc)
    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.char_start < chunk.char_end


def test_empty_document_yields_no_chunks():
    doc = Document(doc_id="d1", text="   ", language="hi", query_type=None)
    for chunker in (FixedWindowChunker(), RecursiveChunker(), MetadataAwareChunker()):
        assert chunker.chunk(doc) == []


def test_fixed_window_overlaps():
    text = " ".join(f"w{i}" for i in range(20))
    doc = Document(doc_id="d1", text=text, language="hi", query_type=None)
    chunks = FixedWindowChunker(window=8, overlap=4).chunk(doc)
    assert len(chunks) > 1
    assert chunks[1].char_start < chunks[0].char_end  # windows actually overlap


def test_fixed_window_rejects_overlap_larger_than_window():
    with pytest.raises(ValueError):
        FixedWindowChunker(window=4, overlap=4)


def test_metadata_chunker_emits_filter_payload():
    doc = Document(doc_id="d1", text=HINDI, language="hi", query_type="DESCRIPTION")
    for chunk in MetadataAwareChunker(target=8).chunk(doc):
        assert chunk.extra["filter_language"] == "hi"
        assert chunk.extra["filter_query_type"] == "DESCRIPTION"


def test_chunk_ids_are_unique_per_strategy():
    doc = Document(doc_id="d1", text=HINDI, language="hi", query_type=None)
    chunks = RecursiveChunker(target=8, overlap=0).chunk(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
