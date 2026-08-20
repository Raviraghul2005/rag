from __future__ import annotations

from pydantic import BaseModel

from app.data.languages import ALL_LANGUAGES, script_fraction

# No ML dependency: script_fraction() (app/data/languages.py, built for the corpus's
# §5 quality gate) already maps each of the 14 languages to its Unicode block. Reused
# here as a ~1ms language-id signal, honest about its real limit — script alone can't
# distinguish languages that share a block (hi/mr/sa/ne all use Devanagari, bn/as both
# use Bengali). "confidence" is the winning script's coverage fraction, not a claim
# about which specific co-script language it is.
SHARED_SCRIPT_GROUPS: dict[str, list[str]] = {
    "hi": ["hi", "mr", "sa", "ne"],
    "mr": ["hi", "mr", "sa", "ne"],
    "sa": ["hi", "mr", "sa", "ne"],
    "ne": ["hi", "mr", "sa", "ne"],
    "bn": ["bn", "as"],
    "as": ["bn", "as"],
}


class LanguageIdResult(BaseModel):
    language: str | None  # best-guess ISO code among the 14, or None if no script matched
    confidence: float  # winning script's coverage fraction of alphabetic characters
    script_ambiguous: bool  # True if the detected script is shared by >1 language
    claimed_language_consistent: bool | None = None  # vs a caller-supplied claim, if any


def identify_language(text: str, claimed_language: str | None = None) -> LanguageIdResult:
    best_lang, best_frac = None, 0.0
    for lang in ALL_LANGUAGES:
        frac = script_fraction(text, lang)
        if frac > best_frac:
            best_lang, best_frac = lang, frac

    ambiguous = best_lang in SHARED_SCRIPT_GROUPS if best_lang else False
    consistent = None
    if claimed_language is not None and best_lang is not None:
        group = SHARED_SCRIPT_GROUPS.get(best_lang, [best_lang])
        consistent = claimed_language in group

    return LanguageIdResult(
        language=best_lang,
        confidence=best_frac,
        script_ambiguous=ambiguous,
        claimed_language_consistent=consistent,
    )
