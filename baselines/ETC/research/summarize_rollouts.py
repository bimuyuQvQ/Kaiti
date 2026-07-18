"""Summarize observed counterfactual benefits without training a controller."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _benefit_bucket(value: float, epsilon: float = 1e-12) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "zero"


def summarize_bundles(bundles: Iterable[Dict[str, Any]], metric: str = "f1") -> Dict[str, Any]:
    bundle_list = list(bundles)
    benefits: List[float] = []
    benefits_by_source: Dict[str, List[float]] = defaultdict(list)
    benefits_by_checkpoint: Dict[str, List[float]] = defaultdict(list)
    bucket_counts = {"positive": 0, "zero": 0, "negative": 0}
    flips = {"wrong_to_correct": 0, "correct_to_wrong": 0}
    state_oracle_gains: List[float] = []
    sample_oracle_gains: List[float] = []
    skip_consistency_diffs: List[float] = []
    total_states = 0
    total_actions = 0

    for bundle in bundle_list:
        states = {state["state_id"]: state for state in bundle["states"]}
        queries = {query["candidate_id"]: query for query in bundle["queries"]}
        actions_by_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for action in bundle["actions"]:
            actions_by_state[action["state_id"]].append(action)
        total_states += len(states)
        total_actions += len(bundle["actions"])
        sample_best = float(bundle["no_retrieval_scores"][metric])

        for state_id, actions in actions_by_state.items():
            skip_rows = [action for action in actions if action["action_type"] == "skip"]
            if len(skip_rows) != 1:
                raise ValueError(f"状态 {state_id} 的 skip 数量不是 1")
            skip = skip_rows[0]
            skip_score = float(skip["scores"][metric])
            baseline_score = float(bundle["no_retrieval_scores"][metric])
            skip_consistency_diffs.append(skip_score - baseline_score)
            state_best = skip_score
            for action in actions:
                if action["action_type"] != "retrieve":
                    continue
                score = float(action["scores"][metric])
                benefit = score - skip_score
                benefits.append(benefit)
                bucket_counts[_benefit_bucket(benefit)] += 1
                query = queries[action["query_candidate_id"]]
                benefits_by_source[query["source"]].append(benefit)
                benefits_by_checkpoint[states[state_id]["checkpoint_type"]].append(benefit)
                if skip["scores"]["accuracy"] == 0 and action["scores"]["accuracy"] == 1:
                    flips["wrong_to_correct"] += 1
                if skip["scores"]["accuracy"] == 1 and action["scores"]["accuracy"] == 0:
                    flips["correct_to_wrong"] += 1
                state_best = max(state_best, score)
                sample_best = max(sample_best, score)
            state_oracle_gains.append(state_best - skip_score)
        sample_oracle_gains.append(sample_best - float(bundle["no_retrieval_scores"][metric]))

    def grouped(values: Dict[str, List[float]]) -> Dict[str, Any]:
        return {
            key: {
                "count": len(group),
                "mean_benefit": _mean(group),
                "positive_rate": sum(value > 0 for value in group) / len(group),
                "negative_rate": sum(value < 0 for value in group) / len(group),
            }
            for key, group in sorted(values.items())
        }

    return {
        "metric": metric,
        "samples": len(bundle_list),
        "states": total_states,
        "actions": total_actions,
        "retrieve_actions": len(benefits),
        "benefit_counts": bucket_counts,
        "mean_retrieval_benefit": _mean(benefits),
        "mean_state_oracle_gain": _mean(state_oracle_gains),
        "mean_sample_timing_query_oracle_gain": _mean(sample_oracle_gains),
        "samples_with_positive_oracle_gain": sum(value > 0 for value in sample_oracle_gains),
        "flips": flips,
        "skip_vs_no_retrieval_max_abs_diff": max(
            (abs(value) for value in skip_consistency_diffs), default=0.0
        ),
        "by_query_source": grouped(benefits_by_source),
        "by_checkpoint_type": grouped(benefits_by_checkpoint),
    }


def load_bundles(run_dir: str | Path) -> List[Dict[str, Any]]:
    paths = sorted((Path(run_dir) / "sample_bundles").glob("sample_*.json"))
    bundles = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            bundles.append(json.load(handle))
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize_bundles(load_bundles(args.run_dir), args.metric)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()

