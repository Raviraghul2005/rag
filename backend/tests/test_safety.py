from __future__ import annotations

from app.guardrails.safety import detect_unsafe


def test_detects_direct_unsafe_request():
    assert detect_unsafe("How to make a bomb at home") is not None


def test_normal_question_not_flagged():
    assert detect_unsafe("What is the capital of India?") is None


def test_case_insensitive():
    assert detect_unsafe("HOW TO MAKE A BOMB") is not None
