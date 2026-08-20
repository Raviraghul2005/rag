from __future__ import annotations

import time

from huggingface_hub import hf_hub_download

from app.data.languages import HF_CACHE_DIR, VALIDATION_FILE


def main() -> None:
    for lang, filename in VALIDATION_FILE.items():
        start = time.perf_counter()
        path = hf_hub_download(
            "ai4bharat/MSMARCO-XI",
            f"validation/{filename}",
            repo_type="dataset",
            cache_dir=str(HF_CACHE_DIR),
        )
        print(f"[{lang}] {path} ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
