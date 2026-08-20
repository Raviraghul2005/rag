from __future__ import annotations

from pydantic import BaseModel


class CorpusPassage(BaseModel):
    passage_id: str
    language: str
    query_type: str | None
    is_selected: bool
    source_query_id: str
    text: str


class EvalQuery(BaseModel):
    query_id: str
    query: str
    language: str
    query_type: str | None
    relevant_passage_ids: list[str]
    # True if this query's relevant passages were deliberately kept out of the corpus —
    # the "corpus-excluded" unanswerable bucket for the abstention curve (spec §11.4).
    held_out: bool
