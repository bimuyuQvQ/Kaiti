"""Compare append and restart outcomes under matched state, query, and evidence."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .summarize_rollouts import _scores, load_bundle_sets


COMPARISON_VERSION = "matched_keep_append_restart_revision_v2"


def _document_signature(action: Mapping[str, Any]) -> List[Tuple[str, int, str]]:
    return [
        (
            str(document.get("document_id") or ""),
            int(document.get("rank", position + 1)),
            str(document.get("text") or ""),
        )
        for position, document in enumerate(action.get("retrieved_documents", []))
    ]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return float(ordered[index])


def _cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    seed: int = 20260722,
    samples: int = 10_000,
) -> Dict[str, float]:
    """Bootstrap sample-level means so query actions are not pseudo-replicates."""

    by_sample: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in rows:
        by_sample[(int(row["sample_index"]), str(row["qid"]))].append(float(row[value_key]))
    cluster_means = [sum(values) / len(values) for values in by_sample.values()]
    if not cluster_means:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "clusters": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(cluster_means) for _ in cluster_means]
        estimates.append(sum(draw) / len(draw))
    return {
        "mean": sum(cluster_means) / len(cluster_means),
        "low": _percentile(estimates, 0.025),
        "high": _percentile(estimates, 0.975),
        "clusters": len(cluster_means),
    }


def _flip(before: float, after: float) -> str:
    if before < 1.0 and after >= 1.0:
        return "wrong_to_correct"
    if before >= 1.0 and after < 1.0:
        return "correct_to_wrong"
    return "unchanged"


def compare_bundle_sets(
    source_bundles: Iterable[Dict[str, Any]],
    restart_bundles: Iterable[Dict[str, Any]],
    revision_bundles: Iterable[Dict[str, Any]] | None = None,
    metric: str = "f1",
    extractor_version: str | None = "first_answer_sentence_v2",
    bootstrap_samples: int = 10_000,
) -> Dict[str, Any]:
    sources = {(int(row["sample_index"]), row["qid"]): row for row in source_bundles}
    restarts = {(int(row["sample_index"]), row["qid"]): row for row in restart_bundles}
    revisions = (
        {(int(row["sample_index"]), row["qid"]): row for row in revision_bundles}
        if revision_bundles is not None
        else None
    )
    if sources.keys() != restarts.keys():
        missing_restart = sorted(set(sources) - set(restarts))
        missing_source = sorted(set(restarts) - set(sources))
        raise ValueError(
            f"源运行与 restart 运行样本不一致：缺 restart={missing_restart[:5]}，缺源={missing_source[:5]}"
        )
    if revisions is not None and sources.keys() != revisions.keys():
        raise ValueError("源运行与 revision 运行样本不一致")

    rows: List[Dict[str, Any]] = []
    no_etc_samples = 0
    for sample_key in sorted(sources):
        source = sources[sample_key]
        restart = restarts[sample_key]
        revision = revisions[sample_key] if revisions is not None else None
        if restart.get("no_etc_trigger"):
            no_etc_samples += 1
            continue
        source_actions = {action["action_id"]: action for action in source["actions"]}
        source_queries = {query["candidate_id"]: query for query in source["queries"]}
        restart_skips = [action for action in restart["actions"] if action["action_type"] == "skip"]
        if len(restart_skips) != 1:
            raise ValueError(f"restart 样本 {sample_key} 的 KEEP 动作数量不是 1")
        keep_action = restart_skips[0]
        keep_score = float(_scores(keep_action, extractor_version)[metric])
        keep_accuracy = float(_scores(keep_action, extractor_version)["accuracy"])
        revision_by_original: Dict[str, Dict[str, Any]] = {}
        if revision is not None:
            revision_by_original = {
                action.get("generation_metadata", {}).get("original_action_id"): action
                for action in revision["actions"]
                if action["action_type"] == "retrieve"
            }

        for restart_action in restart["actions"]:
            if restart_action["action_type"] != "retrieve":
                continue
            metadata = restart_action.get("generation_metadata", {})
            original_action_id = metadata.get("original_action_id")
            if original_action_id not in source_actions:
                raise ValueError(f"restart 动作找不到源 APPEND：{original_action_id}")
            append_action = source_actions[original_action_id]
            if append_action["state_id"] != restart_action["state_id"]:
                raise ValueError(f"状态不匹配：{original_action_id}")
            if _document_signature(append_action) != _document_signature(restart_action):
                raise ValueError(f"检索文档不匹配：{original_action_id}")
            query = source_queries[append_action["query_candidate_id"]]
            append_score = float(_scores(append_action, extractor_version)[metric])
            restart_score = float(_scores(restart_action, extractor_version)[metric])
            append_accuracy = float(_scores(append_action, extractor_version)["accuracy"])
            restart_accuracy = float(_scores(restart_action, extractor_version)["accuracy"])
            operator_scores = {
                "keep": keep_score,
                "append": append_score,
                "restart": restart_score,
            }
            revision_action = revision_by_original.get(original_action_id)
            revision_score = None
            revision_accuracy = None
            revision_metadata: Mapping[str, Any] = {}
            if revision is not None:
                if revision_action is None:
                    raise ValueError(f"revision 动作找不到源 APPEND：{original_action_id}")
                if revision_action["state_id"] != restart_action["state_id"]:
                    raise ValueError(f"revision 状态不匹配：{original_action_id}")
                if _document_signature(append_action) != _document_signature(revision_action):
                    raise ValueError(f"revision 检索文档不匹配：{original_action_id}")
                revision_score = float(_scores(revision_action, extractor_version)[metric])
                revision_accuracy = float(_scores(revision_action, extractor_version)["accuracy"])
                revision_metadata = revision_action.get("generation_metadata", {})
                operator_scores["revision"] = revision_score
            best_score = max(operator_scores.values())
            row = {
                    "sample_index": sample_key[0],
                    "qid": sample_key[1],
                    "state_id": restart_action["state_id"],
                    "query_source": query["source"],
                    "query_text": query["text"],
                    "original_action_id": original_action_id,
                    "restart_action_id": restart_action["action_id"],
                    "document_ids": [item[0] for item in _document_signature(append_action)],
                    "keep_score": keep_score,
                    "append_score": append_score,
                    "restart_score": restart_score,
                    "append_gain_over_keep": append_score - keep_score,
                    "restart_gain_over_keep": restart_score - keep_score,
                    "restart_gain_over_append": restart_score - append_score,
                    "best_operators": sorted(
                        operator for operator, score in operator_scores.items() if score == best_score
                    ),
                    "append_accuracy_flip": _flip(keep_accuracy, append_accuracy),
                    "restart_accuracy_flip": _flip(keep_accuracy, restart_accuracy),
                }
            if revision_score is not None and revision_accuracy is not None:
                row.update(
                    {
                        "revision_action_id": revision_action["action_id"],
                        "revision_score": revision_score,
                        "revision_gain_over_keep": revision_score - keep_score,
                        "revision_gain_over_append": revision_score - append_score,
                        "revision_gain_over_restart": revision_score - restart_score,
                        "revision_accuracy_flip": _flip(keep_accuracy, revision_accuracy),
                        "revision_fallback_to_full_restart": bool(
                            revision_metadata.get("fallback_to_full_restart")
                        ),
                        "revision_rollback_token_index": revision_metadata.get(
                            "rollback_token_index"
                        ),
                    }
                )
            rows.append(row)

    if not rows:
        raise ValueError("没有可比较的同状态干预动作")
    wins = Counter(operator for row in rows for operator in row["best_operators"])
    append_flips = Counter(row["append_accuracy_flip"] for row in rows)
    restart_flips = Counter(row["restart_accuracy_flip"] for row in rows)
    revision_flips = (
        Counter(row["revision_accuracy_flip"] for row in rows) if revisions is not None else None
    )
    preference = Counter(
        "restart_better"
        if row["restart_gain_over_append"] > 0
        else "append_better"
        if row["restart_gain_over_append"] < 0
        else "tie"
        for row in rows
    )
    operators = ["keep", "append", "restart"] + (["revision"] if revisions is not None else [])
    delta_keys = [
        "append_gain_over_keep",
        "restart_gain_over_keep",
        "restart_gain_over_append",
    ]
    if revisions is not None:
        delta_keys.extend(
            [
                "revision_gain_over_keep",
                "revision_gain_over_append",
                "revision_gain_over_restart",
            ]
        )
    report = {
        "comparison_version": COMPARISON_VERSION,
        "metric": metric,
        "extractor_version": extractor_version or "primary",
        "samples": len(sources),
        "samples_without_etc_trigger": no_etc_samples,
        "matched_actions": len(rows),
        "protocol_checks": {
            "same_sample": True,
            "same_state": True,
            "same_query": True,
            "same_retrieved_documents": True,
        },
        "mean_scores": {
            operator: sum(float(row[f"{operator}_score"]) for row in rows) / len(rows)
            for operator in operators
        },
        "paired_deltas": {
            key: _cluster_bootstrap_ci(rows, key, samples=bootstrap_samples)
            for key in delta_keys
        },
        "best_operator_counts_with_ties": dict(sorted(wins.items())),
        "restart_vs_append": dict(sorted(preference.items())),
        "accuracy_flips": {
            "append": dict(sorted(append_flips.items())),
            "restart": dict(sorted(restart_flips.items())),
            **(
                {"revision": dict(sorted(revision_flips.items()))}
                if revision_flips is not None
                else {}
            ),
        },
        "rows": rows,
        "interpretation_caveat": (
            "置信区间按样本聚类后重采样；同一样本的多个查询动作不是独立样本。"
            "revision_strata 将真实句界回退与无可用句界时退化成 full restart 的动作分开报告。"
        ),
    }
    if revisions is not None:
        true_local = [row for row in rows if not row["revision_fallback_to_full_restart"]]
        fallback = [row for row in rows if row["revision_fallback_to_full_restart"]]
        report["revision_strata"] = {
            "true_local_revision": {
                "actions": len(true_local),
                "samples": len({row["sample_index"] for row in true_local}),
                "gain_over_keep": _cluster_bootstrap_ci(
                    true_local, "revision_gain_over_keep", samples=bootstrap_samples
                ),
                "gain_over_append": _cluster_bootstrap_ci(
                    true_local, "revision_gain_over_append", samples=bootstrap_samples
                ),
                "gain_over_restart": _cluster_bootstrap_ci(
                    true_local, "revision_gain_over_restart", samples=bootstrap_samples
                ),
            },
            "fallback_to_full_restart": {
                "actions": len(fallback),
                "samples": len({row["sample_index"] for row in fallback}),
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_dir", required=True, action="append")
    parser.add_argument("--restart_run_dir", required=True, action="append")
    parser.add_argument("--revision_run_dir", action="append", default=None)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--extractor_version", default="first_answer_sentence_v2")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = compare_bundle_sets(
        load_bundle_sets(args.source_run_dir),
        load_bundle_sets(args.restart_run_dir),
        load_bundle_sets(args.revision_run_dir) if args.revision_run_dir else None,
        metric=args.metric,
        extractor_version=args.extractor_version,
        bootstrap_samples=args.bootstrap_samples,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
