"""Deterministic checkpoint selection from an online no-retrieval trace."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import CheckpointState, stable_id


ANSWER_MARKER_RE = re.compile(r"\bthe answer is\b", re.IGNORECASE)
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]?)(?:\s|$)")
CHECKPOINT_VERSION = "checkpoint_selector_v1"


@dataclass(frozen=True)
class TraceObservation:
    generated_prefix: str
    token_index: int
    features: Dict[str, Any] = field(default_factory=dict)
    etc_triggered: bool = False
    etc_query: Optional[str] = None


class CheckpointCollector:
    """Keep the first ETC trigger, first sentence end, and pre-answer state."""

    def __init__(self, qid: str, sample_index: int, max_checkpoints: int = 3) -> None:
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints 必须为正数")
        self.qid = qid
        self.sample_index = sample_index
        self.max_checkpoints = max_checkpoints
        self._candidates: Dict[str, CheckpointState] = {}

    def _add(
        self,
        checkpoint_type: str,
        prefix: str,
        observation: TraceObservation,
        trace_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if checkpoint_type in self._candidates or not prefix.strip():
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "qid": self.qid,
            "sample_index": self.sample_index,
            "checkpoint_type": checkpoint_type,
            "prefix_text": prefix,
        }
        self._candidates[checkpoint_type] = CheckpointState(
            qid=self.qid,
            sample_index=self.sample_index,
            checkpoint_index=-1,
            checkpoint_type=checkpoint_type,
            prefix_text=prefix,
            state_id=stable_id("state", payload),
            token_index=observation.token_index,
            etc_signal=observation.features.get("etc_signal"),
            features=dict(observation.features),
            trace_metadata=dict(trace_metadata or {}),
        )

    def observe(self, observation: TraceObservation) -> None:
        prefix = observation.generated_prefix
        if observation.etc_triggered:
            self._add(
                "first_etc_trigger",
                prefix,
                observation,
                {"etc_query": observation.etc_query},
            )
        boundary = SENTENCE_BOUNDARY_RE.search(prefix)
        if boundary:
            self._add("first_sentence_boundary", prefix[: boundary.end()].rstrip(), observation)
        answer = ANSWER_MARKER_RE.search(prefix)
        if answer:
            self._add("before_first_answer_marker", prefix[: answer.start()].rstrip(), observation)

    def finalize(self) -> List[CheckpointState]:
        unique_by_prefix: Dict[str, CheckpointState] = {}
        priority = {
            "first_etc_trigger": 0,
            "first_sentence_boundary": 1,
            "before_first_answer_marker": 2,
        }
        ordered = sorted(
            self._candidates.values(),
            key=lambda state: (
                state.token_index if state.token_index is not None else 10**12,
                priority[state.checkpoint_type],
            ),
        )
        for state in ordered:
            unique_by_prefix.setdefault(state.prefix_text, state)
        selected = list(unique_by_prefix.values())[: self.max_checkpoints]
        return [
            CheckpointState(
                qid=state.qid,
                sample_index=state.sample_index,
                checkpoint_index=index,
                checkpoint_type=state.checkpoint_type,
                prefix_text=state.prefix_text,
                state_id=state.state_id,
                token_index=state.token_index,
                etc_signal=state.etc_signal,
                features=state.features,
                trace_metadata=state.trace_metadata,
            )
            for index, state in enumerate(selected)
        ]

