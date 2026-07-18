"""Counterfactual action identities, benefit labels, and completeness audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import ActionRollout, stable_id


def make_action_id(state_id: str, action_type: str, query_candidate_id: Optional[str] = None) -> str:
    if action_type not in {"skip", "retrieve"}:
        raise ValueError(f"未知动作类型: {action_type}")
    if (action_type == "retrieve") != bool(query_candidate_id):
        raise ValueError("retrieve 必须关联查询候选，skip 不得关联查询候选")
    return stable_id(
        "act",
        {
            "state_id": state_id,
            "action_type": action_type,
            "query_candidate_id": query_candidate_id,
        },
    )


def counterfactual_benefit(retrieve: ActionRollout, skip: ActionRollout, metric: str = "f1") -> float:
    if retrieve.state_id != skip.state_id:
        raise ValueError("反事实动作必须来自同一状态")
    if retrieve.action_type != "retrieve" or skip.action_type != "skip":
        raise ValueError("动作顺序必须为 retrieve、skip")
    if metric not in retrieve.scores or metric not in skip.scores:
        raise ValueError(f"动作缺少指标: {metric}")
    return float(retrieve.scores[metric] - skip.scores[metric])


@dataclass
class AuditReport:
    expected_states: int
    observed_states: int
    expected_actions: int
    observed_actions: int
    complete: bool
    errors: List[str] = field(default_factory=list)
    benefit_counts: Dict[str, int] = field(default_factory=dict)


def audit_rollouts(
    rollouts: Iterable[ActionRollout],
    expected_queries_by_state: Mapping[str, Sequence[str]],
    metric: str = "f1",
) -> AuditReport:
    rows = list(rollouts)
    errors: List[str] = []
    action_ids = Counter(row.action_id for row in rows)
    for action_id, count in action_ids.items():
        if count > 1:
            errors.append(f"动作重复: {action_id} x {count}")

    grouped: Dict[str, List[ActionRollout]] = defaultdict(list)
    for row in rows:
        grouped[row.state_id].append(row)
        if row.status != "complete":
            errors.append(f"动作未完成: {row.action_id}, status={row.status}")

    expected_actions = 0
    benefit_counts = {"positive": 0, "zero": 0, "negative": 0}
    for state_id, query_ids in expected_queries_by_state.items():
        expected_actions += 1 + len(query_ids)
        state_rows = grouped.get(state_id, [])
        skips = [row for row in state_rows if row.action_type == "skip"]
        if len(skips) != 1:
            errors.append(f"状态 {state_id} 的 skip 数量应为 1，实际为 {len(skips)}")
            continue
        skip = skips[0]
        retrieve_by_query = {
            row.query_candidate_id: row for row in state_rows if row.action_type == "retrieve"
        }
        missing = sorted(set(query_ids) - set(retrieve_by_query))
        extra = sorted(set(retrieve_by_query) - set(query_ids))
        if missing:
            errors.append(f"状态 {state_id} 缺少查询动作: {missing}")
        if extra:
            errors.append(f"状态 {state_id} 出现未声明查询动作: {extra}")
        for query_id in set(query_ids) & set(retrieve_by_query):
            try:
                benefit = counterfactual_benefit(retrieve_by_query[query_id], skip, metric)
            except ValueError as exc:
                errors.append(f"状态 {state_id} 无法计算收益: {exc}")
                continue
            category = "positive" if benefit > 0 else "negative" if benefit < 0 else "zero"
            benefit_counts[category] += 1

    unexpected_states = sorted(set(grouped) - set(expected_queries_by_state))
    if unexpected_states:
        errors.append(f"出现未声明状态: {unexpected_states}")
    return AuditReport(
        expected_states=len(expected_queries_by_state),
        observed_states=len(grouped),
        expected_actions=expected_actions,
        observed_actions=len(rows),
        complete=not errors,
        errors=errors,
        benefit_counts=benefit_counts,
    )

