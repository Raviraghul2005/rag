from __future__ import annotations

import os
from pathlib import Path

# Kept outside the OneDrive-synced repo tree — these are multi-hundred-MB files.
HF_CACHE_DIR = Path(os.environ.get("RAINGOA_HF_CACHE", r"C:\dev-cache\raingoa\hf"))

# The dataset's own loader script is stale (it references .jsonl files that no longer
# exist; the repo holds .parquet), and modern `datasets` won't execute repo scripts
# anyway — so files are addressed directly. Prefixes are the dataset's own 3-letter
# codes, which do not always match the 2-letter language code.
VALIDATION_FILE: dict[str, str] = {
    "as": "asmval.parquet",
    "bn": "benval.parquet",
    "gu": "gujval.parquet",
    "hi": "hinval.parquet",
    "kn": "kanval.parquet",
    "ml": "malval.parquet",
    "mr": "marval.parquet",
    "ne": "nepval.parquet",
    "or": "orival.parquet",
    "pa": "panval.parquet",
    "sa": "sanval.parquet",
    "ta": "tamval.parquet",
    "te": "telval.parquet",
    "ur": "urdval.parquet",
}

# Unicode block each language's script must fall in, for the §5 quality gate.
# Shared blocks are intentional: hi/mr/sa/ne share Devanagari, bn/as share Bengali.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "sa": (0x0900, 0x097F),
    "ne": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "as": (0x0980, 0x09FF),
    "gu": (0x0A80, 0x0AFF),
    "pa": (0x0A00, 0x0A7F),
    "or": (0x0B00, 0x0B7F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "ur": (0x0600, 0x06FF),
}

ALL_LANGUAGES = list(VALIDATION_FILE)


def script_fraction(text: str, lang: str) -> float:
    lo, hi = SCRIPT_RANGES[lang]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)
