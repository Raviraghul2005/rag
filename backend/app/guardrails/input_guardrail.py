from __future__ import annotations

from app.guardrails.injection import detect_injection
from app.guardrails.language_id import identify_language
from app.guardrails.safety import detect_unsafe
from app.models.guardrails import GuardrailOutcome

MIN_TRANSCRIPT_TOKENS = 1  # a bare "?" or single word can still be a valid question


def run_input_guardrails(
    transcript: str, claimed_language: str | None = None
) -> GuardrailOutcome:
    """Pre-retrieval checks (spec §11.1), evaluated in a fixed, cheapest-first order so
    a request that fails early never pays for the later checks."""
    stripped = transcript.strip()
    if not stripped:
        return GuardrailOutcome(allowed=False, reason="empty_transcript")
    if len(stripped.split()) < MIN_TRANSCRIPT_TOKENS:
        return GuardrailOutcome(allowed=False, reason="too_short")

    unsafe_match = detect_unsafe(stripped)
    if unsafe_match:
        return GuardrailOutcome(allowed=False, reason="unsafe_content")

    injection_match = detect_injection(stripped)
    if injection_match:
        return GuardrailOutcome(allowed=False, reason="prompt_injection")

    lang_result = identify_language(stripped, claimed_language)
    if lang_result.language is None:
        # No script in the 14 covered languages matched at all — likely a non-Indic,
        # non-English transcript, or noise. Not necessarily malicious, but outside
        # what this system can serve; logged as its own reason, not lumped in with
        # unsafe/injection.
        return GuardrailOutcome(allowed=False, reason="unrecognized_language")

    return GuardrailOutcome(allowed=True, reason="ok")
