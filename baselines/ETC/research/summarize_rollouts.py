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


def _scores(row: Dict[str, Any], extractor_version: str | None) -> Dict[str, float]:
    if extractor_version is None:
        return row["scores"]
    try:
        return row["alternative_scores"][extractor_version]
    except KeyError as exc:
        raise ValueError(f"记录缺少敏感性抽取器分数: {extractor_version}") from exc


def _baseline_scores(bundle: Dict[str, Any], extractor_version: str | None) -> Dict[str, float]:
    if extractor_version is None:
        return bundle["no_retrieval_scores"]
    try:
        return bundle["no_retrieval_alternative_scores"][extractor_version]
    except KeyError as exc:
        raise ValueError(f"bundle 缺少敏感性抽取器分数: {extractor_version}") from exc


def summarize_bundles(
    bundles: Iterable[Dict[str, Any]],
    metric: str = "f1",
    extractor_version: str | None = None,
    require_skip_consistency: bool = True,
) -> Dict[str, Any]:
    bundle_list = list(bundles)
    benefits: List[float] = []
    benefits_by_source: Dict[str, List[float]] = defaultdict(list)
    benefits_by_checkpoint: Dict[str, List[float]] = defaultdict(list)
    bucket_counts = {"positive": 0, "zero": 0, "negative": 0}
    flips = {"wrong_to_correct": 0, "correct_to_wrong": 0}
    state_oracle_gains: List[float] = []
    sample_oracle_gains: List[float] = []
    skip_consistency_diffs: List[float] = []
    skip_inconsistencies: List[Dict[str, Any]] = []
    nonzero_benefit_cases: List[Dict[str, Any]] = []
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
        baseline_metrics = _baseline_scores(bundle, extractor_version)
        sample_best = float(baseline_metrics[metric])

        for state_id, actions in actions_by_state.items():
            skip_rows = [action for action in actions if action["action_type"] == "skip"]
            if len(skip_rows) != 1:
                raise ValueError(f"状态 {state_id} 的 skip 数量不是 1")
            skip = skip_rows[0]
            skip_metrics = _scores(skip, extractor_version)
            skip_score = float(skip_metrics[metric])
            baseline_score = float(baseline_metrics[metric])
            consistency_diff = skip_score - baseline_score
            skip_consistency_diffs.append(consistency_diff)
            skip_answer = (
                skip["extracted_answer"]
                if extractor_version is None
                else skip["alternative_extractions"][extractor_version]
            )
            baseline_answer = (
                bundle["no_retrieval_extracted_answer"]
                if extractor_version is None
                else bundle["no_retrieval_alternative_extractions"][extractor_version]
            )
            if abs(consistency_diff) > 1e-12 or skip_answer != baseline_answer:
                skip_inconsistencies.append(
                    {
                        "sample_index": bundle["sample_index"],
                        "qid": bundle["qid"],
                        "state_id": state_id,
                        "checkpoint_type": states[state_id]["checkpoint_type"],
                        "no_retrieval_answer": baseline_answer,
                        "no_retrieval_score": baseline_score,
                        "skip_answer": skip_answer,
                        "skip_score": skip_score,
                        "score_diff": consistency_diff,
                    }
                )
            state_best = skip_score
            for action in actions:
                if action["action_type"] != "retrieve":
                    continue
                action_metrics = _scores(action, extractor_version)
                score = float(action_metrics[metric])
                benefit = score - skip_score
                benefits.append(benefit)
                bucket_counts[_benefit_bucket(benefit)] += 1
                query = queries[action["query_candidate_id"]]
                benefits_by_source[query["source"]].append(benefit)
                benefits_by_checkpoint[states[state_id]["checkpoint_type"]].append(benefit)
                if abs(benefit) > 1e-12:
                    nonzero_benefit_cases.append(
                        {
                            "sample_index": bundle["sample_index"],
                            "qid": bundle["qid"],
                            "checkpoint_type": states[state_id]["checkpoint_type"],
                            "query_source": query["source"],
                            "query_text": query["text"],
                            "benefit": benefit,
                            "skip_answer": skip_answer,
                            "skip_score": skip_score,
                            "retrieve_answer": (
                                action["extracted_answer"]
                                if extractor_version is None
                                else action["alternative_extractions"][extractor_version]
                            ),
                            "retrieve_score": score,
                            "injected_sentence": action.get("generation_metadata", {}).get("injected_sentence"),
                        }
                    )
                if skip_metrics["accuracy"] == 0 and action_metrics["accuracy"] == 1:
                    flips["wrong_to_correct"] += 1
                if skip_metrics["accuracy"] == 1 and action_metrics["accuracy"] == 0:
                    flips["correct_to_wrong"] += 1
                state_best = max(state_best, score)
                sample_best = max(sample_best, score)
            state_oracle_gains.append(state_best - skip_score)
        sample_oracle_gains.append(sample_best - float(baseline_metrics[metric]))

    if require_skip_consistency and skip_inconsistencies:
        raise ValueError(
            f"检测到 {len(skip_inconsistencies)} 个 skip/canonical 不一致状态；拒绝计算 oracle"
        )

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
        "extractor_version": extractor_version or "primary",
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
        "skip_inconsistency_count": len(skip_inconsistencies),
        "skip_inconsistencies": skip_inconsistencies,
        "nonzero_benefit_cases": nonzero_benefit_cases,
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
    parser.add_argument("--extractor_version", default=None)
    parser.add_argument("--allow_skip_inconsistency", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize_bundles(
        load_bundles(args.run_dir),
        args.metric,
        extractor_version=args.extractor_version,
        require_skip_consistency=not args.allow_skip_inconsistency,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
