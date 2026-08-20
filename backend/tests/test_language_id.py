from __future__ import annotations

from app.guardrails.language_id import identify_language


def test_identifies_hindi_script():
    result = identify_language("भारत एक विशाल देश है।")
    assert result.language in ("hi", "mr", "sa", "ne")  # Devanagari, genuinely ambiguous
    assert result.script_ambiguous is True
    assert result.confidence > 0.9


def test_identifies_tamil_script_unambiguously():
    result = identify_language("இந்தியா ஒரு பரந்த நாடு.")
    assert result.language == "ta"
    assert result.script_ambiguous is False


def test_no_script_match_returns_none_language():
    result = identify_language("12345 !@#$%")
    assert result.language is None


def test_claimed_language_consistency_flag():
    consistent = identify_language("भारत एक देश है।", claimed_language="hi")
    assert consistent.claimed_language_consistent is True

    inconsistent = identify_language("இந்தியா ஒரு நாடு.", claimed_language="hi")
    assert inconsistent.claimed_language_consistent is False
