from __future__ import annotations

import re

# Same budget and honesty note as injection.py: a curated keyword/regex layer at ~1ms,
# not a trained classifier or LLM judge. Catches direct, unsophisticated requests —
# not obfuscated or adversarially-phrased ones. English-focused; a documented gap for
# the other 13 languages, same as injection.py.
_UNSAFE_PATTERNS = [
    r"\bhow to (make|build|synthesize) (a bomb|explosives?|nerve gas|sarin)\b",
    r"\bhow (do i|to) (kill|murder|poison) (someone|a person|my)\b",
    r"\bhow to (make|cook|synthesize) (meth|methamphetamine|crystal meth)\b",
    r"\b(child|children) (sexual|porn|explicit)\b",
    r"\bhow to (hack|break into) .* (bank account|without permission)\b",
    r"\bself[- ]harm (methods|instructions|how to)\b",
    r"\bhow to (commit|get away with) suicide\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]


def detect_unsafe(text: str) -> str | None:
    """Returns the matched pattern (for logging) or None if nothing matched."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return pattern.pattern
    return None
