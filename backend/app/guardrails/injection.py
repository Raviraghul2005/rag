from __future__ import annotations

import re

# Pattern-based, not model-based: spec §11.1 budgets this stage at ~1ms, and §20
# explicitly rules out LLM-as-judge in the hot path. Groq does host purpose-built
# classifiers for exactly this (meta-llama/llama-prompt-guard-2-22m/-86m) — not used
# here because a network round-trip to them (~100-300ms, measured against this
# project's other Groq calls) would blow the budget by two orders of magnitude. This
# is therefore a real, documented capability limit, not an oversight: a curated
# phrase/regex layer catches unsophisticated attempts and nothing more.
#
# Covers English directly; a handful of common transliterated/native-script variants
# are included for Hindi as a representative sample, not full 14-language coverage —
# stated as a gap in the README rather than silently claimed as solved.
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now (in )?(dan|jailbreak|developer mode)",
    r"act as (if you were|a) (an? )?(unfiltered|unrestricted|jailbroken)",
    r"pretend (you are|to be) (an? )?(ai|assistant) (with no|without) (rules|restrictions|guardrails)",
    r"reveal your (system prompt|instructions|prompt)",
    r"what (is|are) your (system prompt|instructions)",
    r"forget (everything|all) (you|that) (were|was) told",
    r"override (your|the) (system|safety) (prompt|instructions|settings)",
    r"new instructions?:",
    r"system prompt:",
    r"\bdan mode\b",
    r"जो भी निर्देश (दिए गए|मिले) (थे|हैं) उन्हें (भूल|नज़रअंदाज़)",  # "forget/ignore the instructions given"
    r"अपने सिस्टम प्रॉम्प्ट (को )?(बताओ|दिखाओ)",  # "reveal your system prompt"
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> str | None:
    """Returns the matched pattern (for logging) or None if nothing matched."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return pattern.pattern
    return None
