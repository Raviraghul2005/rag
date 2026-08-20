from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.models.retrieval import RetrievalResult


@dataclass
class SpeculationStats:
    """Running counters for the eval package's speculation hit-rate number (spec §9,
    §14.4): "the percentage of queries where the speculative retrieval was reusable,
    and the latency saved." Latency saved is tracked by the caller (it knows the real
    retrieval stage duration whenever a hit lets it skip that stage) — this class only
    counts hits/misses/errors, the inputs to the rate itself.
    """

    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.errors

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class SpeculativeRetrievalCache:
    """Caches retrieval results keyed by a hash of the partial transcript that
    triggered them (spec §9). On the final transcript, `resolve` looks up that exact
    hash: an exact-prefix cache, not a fuzzy/semantic one — the speculative fire-off
    behind this class always uses the latest partial, and Sarvam's partials are
    monotonically-extending prefixes of the eventual final text (verified against
    Sarvam's realtime protocol — transcript.partial always carries straight
    transcription regardless of `mode`), so "does the final transcript equal the last
    partial we fired on" is exactly the reusability question spec §9 asks.

    Speculative work must never block or corrupt the final answer path (spec §9): a
    cache miss or a stored exception both fall through to a normal synchronous
    retrieval at the caller, logged as a miss/error respectively, never raised into the
    final path.
    """

    def __init__(self, min_tokens: int = 3):
        self.min_tokens = min_tokens
        self.stats = SpeculationStats()
        self._cache: dict[str, RetrievalResult | Exception] = {}

    @staticmethod
    def _key(transcript: str) -> str:
        normalized = " ".join(transcript.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def should_speculate(self, partial_transcript: str) -> bool:
        return len(partial_transcript.strip().split()) >= self.min_tokens

    def store(self, transcript: str, result: RetrievalResult | Exception) -> None:
        self._cache[self._key(transcript)] = result

    def resolve(self, final_transcript: str) -> RetrievalResult | None:
        """Returns the cached result on an exact-match hit, or None on a miss — the
        caller runs retrieval normally on None. Updates hit/miss/error stats as a side
        effect of the lookup, since every real final transcript resolves exactly once.
        """
        cached = self._cache.get(self._key(final_transcript))
        if cached is None:
            self.stats.misses += 1
            return None
        if isinstance(cached, Exception):
            self.stats.errors += 1
            return None
        self.stats.hits += 1
        return cached

    def clear(self) -> None:
        """Call once per session/utterance — this cache is keyed by exact text, not
        scoped by session, so stale entries from a previous utterance would otherwise
        accumulate for the lifetime of the process."""
        self._cache.clear()
