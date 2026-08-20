from __future__ import annotations

from app.indexing.sparse_index import BM25SparseIndex, tokenize


def test_tokenize_handles_devanagari_and_lowercases_latin():
    assert tokenize("Hello भारत World") == ["hello", "भारत", "world"]


def test_search_ranks_exact_term_match_above_unrelated_doc():
    index = BM25SparseIndex()
    index.fit(
        [
            "the quick brown fox jumps over the lazy dog",
            "completely unrelated text about something else entirely",
            "a fox is a small quick animal",
        ]
    )
    results = index.search("quick fox", top_k=3)
    assert results  # at least one match
    ranked_rows = [row for row, _ in results]
    assert ranked_rows[0] in (0, 2)  # docs containing "quick"/"fox" outrank doc 1
    assert 1 not in ranked_rows[:1]


def test_search_with_no_vocabulary_overlap_returns_empty():
    index = BM25SparseIndex()
    index.fit(["alpha beta gamma"])
    assert index.search("zzz nonexistent", top_k=5) == []


def test_save_and_load_round_trip(tmp_path):
    index = BM25SparseIndex()
    index.fit(["quick fox", "slow turtle", "quick turtle"])
    before = index.search("quick", top_k=3)

    index.save(tmp_path / "sparse")
    reloaded = BM25SparseIndex.load(tmp_path / "sparse")
    after = reloaded.search("quick", top_k=3)

    assert before == after
