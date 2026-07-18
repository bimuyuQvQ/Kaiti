"""Deterministic checkpoint selection from an online no-retrieval trace."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import CheckpointState, stable_id


ANSWER_MARKER_RE = re.compile(r"\bthe answer is\b", re.IGNORECASE)
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]?)(?:\s|$)")
CHECKPOINT_VERSION = "checkpoint_selector_v3"


@dataclass(frozen=True)
class TraceObservation:
    generated_prefix: str
    token_index: int
    features: Dict[str, Any] = field(default_factory=dict)
    etc_triggered: bool = False
    etc_query: Optional[str] = None
    prefix_token_ids: List[int] = field(default_factory=list)


class CheckpointCollector:
    """Collect legacy anchors or ETC-independent dense timing candidates."""

    def __init__(
        self,
        qid: str,
        sample_index: int,
        max_checkpoints: int = 3,
        timing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints 必须为正数")
        self.qid = qid
        self.sample_index = sample_index
        self.max_checkpoints = max_checkpoints
        self.timing_config = dict(timing_config or {})
        self.mode = self.timing_config.get("mode", "legacy_anchors_v2")
        if self.mode not in {"legacy_anchors_v2", "dense_timing_v1"}:
            raise ValueError(f"未知时机候选模式: {self.mode}")
        self.token_stride = int(self.timing_config.get("token_stride", 0))
        self.max_token_grid = int(self.timing_config.get("max_token_grid_checkpoints", 0))
        self.include_etc_trigger = bool(self.timing_config.get("include_etc_trigger", True))
        self.include_sentence_boundaries = bool(
            self.timing_config.get("include_sentence_boundaries", True)
        )
        self.include_before_answer = bool(
            self.timing_config.get("include_before_answer", True)
        )
        self.selection_policy = self.timing_config.get(
            "selection_policy",
            "temporal_stratified_preserve_anchors_v1",
        )
        if (
            self.mode == "dense_timing_v1"
            and self.selection_policy != "temporal_stratified_preserve_anchors_v1"
        ):
            raise ValueError(f"未知密集时机预算策略: {self.selection_policy}")
        if self.mode == "dense_timing_v1" and self.token_stride <= 0:
            raise ValueError("dense_timing_v1 的 token_stride 必须为正数")
        if self.max_token_grid < 0:
            raise ValueError("max_token_grid_checkpoints 不能为负数")
        self._candidates: Dict[str, CheckpointState] = {}
        self._history: List[TraceObservation] = []
        self._observed_sentence_boundaries = 0
        self._next_token_milestone = self.token_stride
        self._token_grid_count = 0
        self._answer_marker_seen = False

    def _add(
        self,
        checkpoint_type: str,
        prefix: str,
        observation: TraceObservation,
        trace_metadata: Optional[Dict[str, Any]] = None,
        candidate_key: Optional[str] = None,
    ) -> None:
        key = candidate_key or checkpoint_type
        if key in self._candidates or not prefix.strip():
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "qid": self.qid,
            "sample_index": self.sample_index,
            "checkpoint_type": checkpoint_type,
            "candidate_key": key,
            "prefix_text": prefix,
            "prefix_token_ids": observation.prefix_token_ids,
        }
        self._candidates[key] = CheckpointState(
            qid=self.qid,
            sample_index=self.sample_index,
            checkpoint_index=-1,
            checkpoint_type=checkpoint_type,
            prefix_text=prefix,
            prefix_token_ids=list(observation.prefix_token_ids),
            state_id=stable_id("state", payload),
            token_index=observation.token_index,
            etc_signal=observation.features.get("etc_signal"),
            features={
                **dict(observation.features),
                "generated_token_index": observation.token_index,
                "generated_character_count": len(prefix),
                "sentence_boundaries_seen": len(
                    list(SENTENCE_BOUNDARY_RE.finditer(prefix))
                ),
            },
            trace_metadata={
                "prefix_token_source": "legacy_generate_online_exact_ids_v2",
                "timing_candidate_mode": self.mode,
                "timing_selection_policy": self.selection_policy,
                "candidate_key": key,
                **dict(trace_metadata or {}),
            },
        )

    def observe(self, observation: TraceObservation) -> None:
        prefix = observation.generated_prefix
        answer = ANSWER_MARKER_RE.search(prefix)
        if self.mode == "dense_timing_v1" and (answer or self._answer_marker_seen):
            if answer and not self._answer_marker_seen and self.include_before_answer:
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
            self._answer_marker_seen = True
            self._history.append(observation)
            return
        if observation.etc_triggered and self.include_etc_trigger:
            self._add(
                "first_etc_trigger",
                prefix,
                observation,
                {"etc_query": observation.etc_query},
            )
        boundary_count = len(list(SENTENCE_BOUNDARY_RE.finditer(prefix)))
        if self.mode == "dense_timing_v1":
            while (
                self.include_sentence_boundaries
                and self._observed_sentence_boundaries < boundary_count
            ):
                self._observed_sentence_boundaries += 1
                occurrence = self._observed_sentence_boundaries
                self._add(
                    "sentence_boundary",
                    prefix,
                    observation,
                    {"sentence_boundary_index": occurrence},
                    candidate_key=f"sentence_boundary_{occurrence}",
                )
            while (
                self._token_grid_count < self.max_token_grid
                and observation.token_index >= self._next_token_milestone
            ):
                milestone = self._next_token_milestone
                self._token_grid_count += 1
                self._next_token_milestone += self.token_stride
                self._add(
                    "token_grid",
                    prefix,
                    observation,
                    {"token_milestone": milestone},
                    candidate_key=f"token_grid_{milestone}",
                )
        elif boundary_count and self.include_sentence_boundaries:
            # 以首次检测到句界时的真实 token 状态为检查点，避免把字符串
            # 截断到一个无法由原 token 前缀精确表示的位置。
            self._add("first_sentence_boundary", prefix, observation)
        if answer and self.include_before_answer:
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
            "sentence_boundary": 1,
            "first_sentence_boundary": 1,
            "token_grid": 2,
            "before_first_answer_marker": 3,
        }
        ordered = sorted(
            self._candidates.values(),
            key=lambda state: (
                state.token_index if state.token_index is not None else 10**12,
                priority.get(state.checkpoint_type, 99),
            ),
        )
        for state in ordered:
            unique_by_prefix.setdefault(tuple(state.prefix_token_ids), state)
        unique = list(unique_by_prefix.values())
        if self.mode == "dense_timing_v1":
            selected = self._temporally_stratified(unique)
        else:
            selected = unique[: self.max_checkpoints]
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

    def _temporally_stratified(self, ordered: List[CheckpointState]) -> List[CheckpointState]:
        """Preserve anchor states, then cover the remaining timeline evenly."""

        if len(ordered) <= self.max_checkpoints:
            return ordered
        anchors: List[CheckpointState] = []
        for checkpoint_type in ("first_etc_trigger", "before_first_answer_marker"):
            match = next((state for state in ordered if state.checkpoint_type == checkpoint_type), None)
            if match is not None and match not in anchors:
                anchors.append(match)
        if len(anchors) >= self.max_checkpoints:
            return sorted(anchors[: self.max_checkpoints], key=lambda state: state.token_index or 0)

        pool = [state for state in ordered if state not in anchors]
        slots = self.max_checkpoints - len(anchors)
        if len(pool) <= slots:
            chosen = pool
        elif slots == 1:
            chosen = [pool[len(pool) // 2]]
        else:
            indices = {
                round(position * (len(pool) - 1) / (slots - 1))
                for position in range(slots)
            }
            chosen = [pool[index] for index in sorted(indices)]
            if len(chosen) < slots:
                chosen_ids = {state.state_id for state in chosen}
                chosen.extend(
                    state for state in pool if state.state_id not in chosen_ids
                )
                chosen = chosen[:slots]
        return sorted(anchors + chosen, key=lambda state: state.token_index or 0)
