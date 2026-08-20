from __future__ import annotations

# Maps the corpus's 2-letter language codes (app/data/languages.py's ALL_LANGUAGES,
# from the ai4bharat/MSMARCO-XI dataset's own convention) to Sarvam Saaras v3's BCP-47
# codes (verified against Sarvam's live docs, 2026-08-20 — see docs.sarvam.ai/api-
# reference-docs/models/saaras). Almost all follow "{code}-IN", but Odia is the one
# exception: the corpus uses ISO 639-1 "or", Sarvam uses "od-IN" — a real mismatch,
# not a typo, so it needs an explicit table entry rather than a format-string rule.
#
# Spec §9 warned that "Sarvam STT does not cover all [14] of the dataset's languages"
# and asked for that gap to be stated explicitly. As of this build, that gap does not
# exist: Sarvam's saaras:v3-realtime covers all 14 (Sanskrit and Odia included, per
# the same source), so every corpus language has a live streaming STT path. Worth
# recording as an update to the spec's assumption rather than silently dropping the
# warning the spec asked for.
CORPUS_TO_SARVAM: dict[str, str] = {
    "as": "as-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "or": "od-IN",  # exception: corpus code "or" (ISO 639-1) vs Sarvam's "od-IN"
    "pa": "pa-IN",
    "sa": "sa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN",
}

SARVAM_TO_CORPUS: dict[str, str] = {v: k for k, v in CORPUS_TO_SARVAM.items()}


def to_sarvam_code(corpus_language: str) -> str:
    try:
        return CORPUS_TO_SARVAM[corpus_language]
    except KeyError:
        raise ValueError(f"no Sarvam language code mapping for {corpus_language!r}") from None


def to_corpus_code(sarvam_language: str) -> str | None:
    """Returns None for a Sarvam code outside the corpus's 14 languages (e.g. en-IN,
    kok-IN) — a valid outcome, not an error, since Sarvam covers more languages than
    the corpus does."""
    return SARVAM_TO_CORPUS.get(sarvam_language)
