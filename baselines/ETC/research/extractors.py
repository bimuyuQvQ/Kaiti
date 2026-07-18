"""Versioned answer extraction rules.

`first_answer_span_v1` is an exact research-layer copy of the corrected rule in
`evaluate_online.py`.  Keeping a named copy makes labels reproducible even when
the legacy evaluator changes later.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


EXTRACTOR_VERSION = "first_answer_span_v1"
SENSITIVITY_EXTRACTOR_VERSION = "first_answer_sentence_v2"
ANSWER_MARKER_RE = re.compile(r"\bthe answer is\b", re.IGNORECASE)
REPEATED_ANSWER_RE = re.compile(r"\b(?:so\s+)?the answer is\b", re.IGNORECASE)
QUESTION_TAIL_RE = re.compile(r"(?:(?<=\s)|(?<=\.))Question\b", re.IGNORECASE)
SPLIT_WORDS = ["Question:", "#10000000", "Note:"]


def strip_generation_tail(text: str) -> str:
    for word in SPLIT_WORDS:
        position = text.find(word)
        if position != -1 and position > 0:
            text = text[:position]
    return text


def extract_first_answer_span_v1(text: str) -> str:
    text = strip_generation_tail(text)
    match = ANSWER_MARKER_RE.search(text)
    if not match:
        return ""
    text = text[match.end() :].strip()

    repeated = REPEATED_ANSWER_RE.search(text)
    if repeated:
        text = text[: repeated.start()].strip()
    question_tail = QUESTION_TAIL_RE.search(text)
    if question_tail:
        text = text[: question_tail.start()].strip()

    for suffix in ["</s>", "<|endoftext|>"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def extract_first_answer_sentence_v2(text: str) -> str:
    """Sensitivity view that removes answer-following explanations.

    The primary v1 rule remains frozen.  This rule is deliberately small and
    deterministic: yes/no answers collapse to the leading polarity token;
    otherwise only the first sentence after the answer marker is retained.
    """

    answer = extract_first_answer_span_v1(text)
    polarity = re.match(r"^(yes|no)\b", answer, re.IGNORECASE)
    if polarity:
        return polarity.group(1)
    boundary = re.search(r"[.!?](?=\s+(?:[A-Z0-9]|However\b|Both\b|Because\b))", answer)
    if boundary:
        answer = answer[: boundary.start()].strip()
    return answer


def extract_answer(
    text: str,
    mode: str = EXTRACTOR_VERSION,
    legacy_extractor: Optional[Callable[[str], str]] = None,
) -> str:
    if mode == "first_answer_span_v1":
        return extract_first_answer_span_v1(text)
    if mode == "first_answer_sentence_v2":
        return extract_first_answer_sentence_v2(text)
    if mode == "raw_v1":
        return strip_generation_tail(text).strip()
    if mode == "legacy_original":
        if legacy_extractor is None:
            raise ValueError("legacy_original 模式必须显式传入 legacy_extractor")
        return legacy_extractor(strip_generation_tail(text))
    raise ValueError(f"未知答案抽取模式: {mode}")
