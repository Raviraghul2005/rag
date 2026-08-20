from __future__ import annotations

import app.env_bootstrap  # noqa: F401 — must run before anything reads os.environ or HF_HOME

import time

from app.indexing.embeddings import E5Encoder

# A few real sentences per language (not lorem ipsum) so tokenization cost reflects
# actual Indic script behavior, not just Latin text padded out.
SAMPLE_TEXTS = [
    "भारत एक विशाल और विविधतापूर्ण देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "ভারত একটি বিশাল ও বৈচিত্র্যময় দেশ যেখানে অনেক ভাষায় কথা বলা হয়।",
    "இந்தியா ஒரு பரந்த மற்றும் பன்முகத்தன்மை கொண்ட நாடாகும்.",
    "భారతదేశం అనేక భాషలు మాట్లాడే విశాలమైన మరియు వైవిధ్యభరితమైన దేశం.",
    "India is a vast and diverse country where many languages are spoken.",
] * 20  # 100 texts total


def main() -> None:
    print("Loading / quantizing encoder (one-time cost, not part of the throughput number)...")
    setup_start = time.perf_counter()
    encoder = E5Encoder()
    setup_s = time.perf_counter() - setup_start
    print(f"Setup took {setup_s:.1f}s")

    # Warm-up run — first inference pays a one-time graph-optimization cost.
    encoder.encode_passages(SAMPLE_TEXTS[:8])

    start = time.perf_counter()
    encoder.encode_passages(SAMPLE_TEXTS)
    elapsed_s = time.perf_counter() - start

    throughput = len(SAMPLE_TEXTS) / elapsed_s
    print(f"\nEncoded {len(SAMPLE_TEXTS)} passages in {elapsed_s:.2f}s")
    print(f"Throughput: {throughput:.1f} passages/sec (CPU, e5-small int8, batch={len(SAMPLE_TEXTS)})")

    # N-calibration per the plan: pick N so a full 14-lang x 5-strategy indexing run
    # stays within a modest, single unattended pass. Conservative avg 1.3 chunks/passage.
    languages, strategies, avg_chunks_per_passage = 14, 5, 1.3
    for budget_hours in (0.5, 1, 2):
        n = int((budget_hours * 3600 * throughput) / (languages * strategies * avg_chunks_per_passage))
        print(f"  at {budget_hours}h dense-encode budget -> N ~= {n:,} passages/language")


if __name__ == "__main__":
    main()
