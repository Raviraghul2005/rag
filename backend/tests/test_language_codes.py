from __future__ import annotations

import pytest

from app.data.languages import ALL_LANGUAGES
from app.stt.language_codes import CORPUS_TO_SARVAM, to_corpus_code, to_sarvam_code


def test_every_corpus_language_has_a_sarvam_mapping():
    for lang in ALL_LANGUAGES:
        assert lang in CORPUS_TO_SARVAM


def test_odia_uses_the_od_exception_not_or():
    assert to_sarvam_code("or") == "od-IN"


def test_regular_mapping_follows_in_suffix():
    assert to_sarvam_code("hi") == "hi-IN"
    assert to_sarvam_code("ta") == "ta-IN"


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        to_sarvam_code("xx")


def test_round_trip_corpus_to_sarvam_to_corpus():
    for lang in ALL_LANGUAGES:
        assert to_corpus_code(to_sarvam_code(lang)) == lang


def test_sarvam_only_language_maps_to_none():
    assert to_corpus_code("en-IN") is None
    assert to_corpus_code("kok-IN") is None
