"""Auditable query candidates shared by oracle collection and learned policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .schema import QueryCandidate, stable_id


QUERY_SET_VERSION = "query_set_v1"


@dataclass(frozen=True)
class QueryContext:
    qid: str
    state_id: str
    question: str
    prefix_text: str
    etc_query: Optional[str] = None


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def prefix_gap_prompt(context: QueryContext) -> str:
    """Build the fixed prompt; model invocation belongs to the rollout worker."""

    return (
        "Given the question and the partial reasoning, write one short search query "
        "for the single missing fact needed next. Do not answer the question.\n"
        f"Question: {context.question.strip()}\n"
        f"Partial reasoning: {context.prefix_text.strip()}\n"
        "Search query:"
    )


def _candidate(context: QueryContext, source: str, text: str, metadata: Dict[str, Any]) -> QueryCandidate:
    normalized = normalize_query(text)
    payload = {
        "version": QUERY_SET_VERSION,
        "qid": context.qid,
        "state_id": context.state_id,
        "source": source,
        "normalized_text": normalized,
    }
    return QueryCandidate(
        qid=context.qid,
        state_id=context.state_id,
        source=source,
        text=text.strip(),
        normalized_text=normalized,
        candidate_id=stable_id("qry", payload),
        metadata=metadata,
    )


def build_query_candidates(
    context: QueryContext,
    generated_prefix_gap_query: Optional[str] = None,
) -> List[QueryCandidate]:
    """Build and de-duplicate question, ETC-QFS, and prefix-gap candidates."""

    raw = [("question", context.question, {})]
    if context.etc_query and context.etc_query.strip():
        raw.append(("etc_qfs", context.etc_query, {}))
    if generated_prefix_gap_query and generated_prefix_gap_query.strip():
        raw.append(
            (
                "prefix_gap_v1",
                generated_prefix_gap_query,
                {"prompt_version": "prefix_gap_prompt_v1"},
            )
        )

    seen = set()
    candidates: List[QueryCandidate] = []
    for source, text, metadata in raw:
        normalized = normalize_query(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(_candidate(context, source, text, metadata))
    return candidates


def assert_unique_candidates(candidates: Iterable[QueryCandidate]) -> None:
    ids = set()
    normalized = set()
    for candidate in candidates:
        if candidate.candidate_id in ids:
            raise ValueError(f"查询候选 ID 重复: {candidate.candidate_id}")
        key = (candidate.state_id, candidate.normalized_text)
        if key in normalized:
            raise ValueError(f"同一状态存在等价查询: {candidate.normalized_text}")
        ids.add(candidate.candidate_id)
        normalized.add(key)

