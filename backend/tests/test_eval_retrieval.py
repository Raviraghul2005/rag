from __future__ import annotations

from scripts.eval_retrieval import _source_doc_id


def test_strips_strategy_and_index_suffix():
    assert _source_doc_id("hi_1102432_5::recursive_512::0") == "hi_1102432_5"


def test_strategy_name_containing_no_extra_colons():
    assert _source_doc_id("hi_42_0::fixed_256_overlap_64::12") == "hi_42_0"


def test_matches_relevant_passage_id_format():
    # relevant_passage_ids in the eval set are bare passage_ids (build_corpus.py) —
    # this is the exact real-world pairing that was silently never matching before
    # the fix (every query scored a total miss regardless of actual retrieval quality).
    chunk_id = "hi_1102432_5::recursive_512::0"
    relevant_passage_id = "hi_1102432_5"
    assert _source_doc_id(chunk_id) == relevant_passage_id
