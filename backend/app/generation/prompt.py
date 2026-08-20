from __future__ import annotations

from app.chunking.base import word_spans
from app.models.retrieval import ScoredChunk

SYSTEM_PROMPT = """You are a retrieval-grounded question answering system.

Rules:
- Answer only using the provided context passages. Never use outside knowledge, even if you know the answer.
- Answer in the same language as the question.
- List the chunk_id of every passage you actually used in cited_chunk_ids.
- If the context does not contain enough information to answer the question, set "sufficient" to false, "answer" to null, and "cited_chunk_ids" to an empty list. Do not guess.
- Respond with ONLY a JSON object of this exact shape, no markdown fences, no extra text:
{"answer": "<answer text or null>", "cited_chunk_ids": ["<chunk_id>", ...], "sufficient": <true or false>}"""


def trim_context(chunks: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
    """Greedy-fills the context window in ranking order, stopping once the whitespace-
    word count (the same cheap token proxy app/chunking/base.py uses for chunk sizing —
    consistent budget accounting across the codebase, not exact target-model token
    counts) would exceed the budget. Always keeps at least the top-ranked chunk, even
    if it alone exceeds the budget, so a too-tight budget degrades gracefully rather
    than emptying the context entirely.
    """
    kept: list[ScoredChunk] = []
    used = 0
    for chunk in chunks:
        length = len(word_spans(chunk.text))
        if kept and used + length > token_budget:
            break
        kept.append(chunk)
        used += length
        if used >= token_budget:
            break
    return kept


def build_messages(question: str, context: list[ScoredChunk]) -> list[dict]:
    if not context:
        context_block = "(no context passages retrieved)"
    else:
        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context)
    user_prompt = f"Context passages:\n{context_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
