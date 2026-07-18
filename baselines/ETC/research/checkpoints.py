"""Deterministic checkpoint selection from an online no-retrieval trace."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import CheckpointState, stable_id


ANSWER_MARKER_RE = re.compile(r"\bthe answer is\b", re.IGNORECASE)
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]?)(?:\s|$)")
CHECKPOINT_VERSION = "checkpoint_selector_v2"


@dataclass(frozen=True)
class TraceObservation:
    generated_prefix: str
    token_index: int
    features: Dict[str, Any] = field(default_factory=dict)
    etc_triggered: bool = False
    etc_query: Optional[str] = None
    prefix_token_ids: List[int] = field(default_factory=list)


class CheckpointCollector:
    """Keep the first ETC trigger, first sentence end, and pre-answer state."""

    def __init__(self, qid: str, sample_index: int, max_checkpoints: int = 3) -> None:
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints 必须为正数")
        self.qid = qid
        self.sample_index = sample_index
        self.max_checkpoints = max_checkpoints
        self._candidates: Dict[str, CheckpointState] = {}
        self._history: List[TraceObservation] = []

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
            "prefix_token_ids": observation.prefix_token_ids,
        }
        self._candidates[checkpoint_type] = CheckpointState(
            qid=self.qid,
            sample_index=self.sample_index,
            checkpoint_index=-1,
            checkpoint_type=checkpoint_type,
            prefix_text=prefix,
            prefix_token_ids=list(observation.prefix_token_ids),
            state_id=stable_id("state", payload),
            token_index=observation.token_index,
            etc_signal=observation.features.get("etc_signal"),
            features=dict(observation.features),
            trace_metadata={
                "prefix_token_source": "legacy_generate_online_exact_ids_v2",
                **dict(trace_metadata or {}),
            },
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
            # 以首次检测到句界时的真实 token 状态为检查点，避免把字符串
            # 截断到一个无法由原 token 前缀精确表示的位置。
            self._add("first_sentence_boundary", prefix, observation)
        answer = ANSWER_MARKER_RE.search(prefix)
        if answer:
            # 回看逐 token 观测，选择答案标记出现之前最后一个真实状态。
            prior = [
                item
                for item in self._history
                if len(item.generated_prefix.rstrip()) <= answer.start()
            ]
            if prior:
                previous = prior[-1]
                self._add(
                    "before_first_answer_marker",
                    previous.generated_prefix,
                    previous,
                )
        self._history.append(observation)

    def finalize(self) -> List[CheckpointState]:
        unique_by_prefix: Dict[tuple[int, ...], CheckpointState] = {}
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
            unique_by_prefix.setdefault(tuple(state.prefix_token_ids), state)
        selected = list(unique_by_prefix.values())[: self.max_checkpoints]
        return [
            CheckpointState(
                qid=state.qid,
                sample_index=state.sample_index,
                checkpoint_index=index,
                checkpoint_type=state.checkpoint_type,
                prefix_text=state.prefix_text,
                prefix_token_ids=state.prefix_token_ids,
                state_id=state.state_id,
                token_index=state.token_index,
                etc_signal=state.etc_signal,
                features=state.features,
                trace_metadata=state.trace_metadata,
            )
            for index, state in enumerate(selected)
        ]
