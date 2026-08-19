from __future__ import annotations

from pydantic import BaseModel


class GuardrailOutcome(BaseModel):
    allowed: bool
    reason: str
