"""Nested group-CV probe for learning KEEP versus FULL_RESTART decisions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROBE_VERSION = "keep_restart_nested_group_probe_v1"


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "ground_truth" in row or "extracted_answer" in row:
                raise ValueError(f"第 {line_number} 行包含禁止的答案字段")
            rows.append(row)
    if not rows:
        raise ValueError("监督文件为空")
    return rows


def _numeric_keys(rows: Sequence[Mapping[str, Any]], variant: str) -> List[str]:
    keys = sorted({key for row in rows for key in row["visible_numeric"]})
    if variant == "no_prefix_tfidf":
        keys = [
            key
            for key in keys
            if not key.startswith("prefix_")
            and not key.startswith("state_")
            and key != "query_source_prefix_gap_v1"
        ]
    return keys


def _matrices(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    variant: str,
):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    keys = _numeric_keys(train_rows, variant)
    train_numeric = np.asarray(
        [[float(row["visible_numeric"].get(key, 0.0)) for key in keys] for row in train_rows],
        dtype=float,
    )
    test_numeric = np.asarray(
        [[float(row["visible_numeric"].get(key, 0.0)) for key in keys] for row in test_rows],
        dtype=float,
    )
    scaler = StandardScaler()
    train_numeric = csr_matrix(scaler.fit_transform(train_numeric))
    test_numeric = csr_matrix(scaler.transform(test_numeric))
    if variant == "numeric":
        return train_numeric, test_numeric, {"numeric_features": len(keys), "tfidf_features": 0}

    text_key = "text_full" if variant == "full_tfidf" else "text_no_prefix"
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    try:
        train_text = vectorizer.fit_transform([str(row[text_key]) for row in train_rows])
    except ValueError:
        vectorizer.set_params(min_df=1)
        train_text = vectorizer.fit_transform([str(row[text_key]) for row in train_rows])
    test_text = vectorizer.transform([str(row[text_key]) for row in test_rows])
    return (
        hstack([train_numeric, train_text], format="csr"),
        hstack([test_numeric, test_text], format="csr"),
        {"numeric_features": len(keys), "tfidf_features": len(vectorizer.vocabulary_)},
    )


def _fit_predict(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    variant: str,
) -> Tuple[List[float], List[float], Dict[str, int]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge

    x_train, x_test, dimensions = _matrices(train_rows, test_rows, variant)
    delta_train = np.asarray(
        [float(row["labels"]["restart_gain_over_keep"]) for row in train_rows], dtype=float
    )
    harm_train = np.asarray(
        [int(row["labels"]["restart_harm"]) for row in train_rows], dtype=int
    )
    value_model = Ridge(alpha=10.0, solver="lsqr")
    value_model.fit(x_train, delta_train)
    delta_prediction = value_model.predict(x_test)
    if len(set(harm_train.tolist())) < 2:
        harm_prediction = np.full(len(test_rows), float(harm_train[0]), dtype=float)
    else:
        harm_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=20260730,
        )
        harm_model.fit(x_train, harm_train)
        harm_prediction = harm_model.predict_proba(x_test)[:, 1]
    return delta_prediction.tolist(), harm_prediction.tolist(), dimensions


def _group_splits(rows: Sequence[Mapping[str, Any]], folds: int):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([str(row["qid"]) for row in rows])
    unique_groups = len(set(groups.tolist()))
    n_splits = min(folds, unique_groups)
    if n_splits < 2:
        raise ValueError("至少需要两个 qid 才能做 group CV")
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(rows))
    return list(splitter.split(dummy, groups=groups))


def _inner_predictions(
    rows: Sequence[Mapping[str, Any]], variant: str, folds: int
) -> Tuple[List[float], List[float]]:
    delta = [0.0] * len(rows)
    harm = [0.0] * len(rows)
    for train_indices, validation_indices in _group_splits(rows, folds):
        train_rows = [rows[int(index)] for index in train_indices]
        validation_rows = [rows[int(index)] for index in validation_indices]
        delta_pred, harm_pred, _ = _fit_predict(train_rows, validation_rows, variant)
        for index, prediction, risk in zip(validation_indices, delta_pred, harm_pred):
            delta[int(index)] = float(prediction)
            harm[int(index)] = float(risk)
    return delta, harm


def _choose_policy(
    rows: Sequence[Mapping[str, Any]],
    delta_prediction: Sequence[float],
    harm_prediction: Sequence[float],
    risk_aware: bool,
) -> Dict[str, float]:
    import numpy as np

    lambdas = [0.0] if not risk_aware else [0.0, 0.1, 0.25, 0.5, 1.0]
    harm_penalty = 0.0 if not risk_aware else 0.10
    best: Tuple[float, float, float, float] | None = None
    for weight in lambdas:
        adjusted = np.asarray(delta_prediction) - weight * np.asarray(harm_prediction)
        thresholds = set(
            float(value)
            for value in np.quantile(adjusted, np.linspace(0.0, 1.0, 41)).tolist()
        )
        thresholds.update({0.0, float(adjusted.min() - 1e-9), float(adjusted.max() + 1e-9)})
        for threshold in sorted(thresholds):
            restart = adjusted > threshold
            selected_scores = [
                float(row["labels"]["restart_f1"])
                if choose_restart
                else float(row["labels"]["keep_f1"])
                for row, choose_restart in zip(rows, restart)
            ]
            harms = [
                int(row["labels"]["restart_harm"]) if choose_restart else 0
                for row, choose_restart in zip(rows, restart)
            ]
            score = float(np.mean(selected_scores))
            harm_rate = float(np.mean(harms))
            objective = score - harm_penalty * harm_rate
            candidate = (objective, score, -harm_rate, -abs(float(threshold)))
            if best is None or candidate > best:
                best = candidate
                best_weight = float(weight)
                best_threshold = float(threshold)
    assert best is not None
    return {
        "lambda_harm": best_weight,
        "threshold": best_threshold,
        "inner_objective": best[0],
        "inner_selected_f1": best[1],
        "inner_harm_rate": -best[2],
        "harm_penalty": harm_penalty,
    }


def _cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
    seed: int = 20260730,
    samples: int = 10_000,
) -> Dict[str, float]:
    by_qid: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        by_qid[str(record["qid"])].append(float(record[value_key]))
    values = [sum(items) / len(items) for items in by_qid.values()]
    rng = random.Random(seed)
    estimates = [
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    ]
    estimates.sort()
    return {
        "mean": sum(values) / len(values),
        "low": estimates[int(0.025 * len(estimates))],
        "high": estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))],
        "qids": len(values),
    }


def _aggregate(
    records: Sequence[Mapping[str, Any]], policy: str, bootstrap_samples: int
) -> Dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    selected_key = f"{policy}_selected_score"
    action_key = f"{policy}_selected_action"
    selected = np.asarray([float(record[selected_key]) for record in records])
    fixed = np.asarray([float(record["fixed_score"]) for record in records])
    oracle = np.asarray([float(record["oracle_score"]) for record in records])
    denominator = float(np.mean(oracle) - np.mean(fixed))
    harms = np.asarray(
        [
            int(record["restart_harm"]) if record[action_key] == "restart" else 0
            for record in records
        ]
    )
    rescues = np.asarray(
        [
            int(record["restart_rescue"]) if record[action_key] == "restart" else 0
            for record in records
        ]
    )
    harm_labels = np.asarray([int(record["restart_harm"]) for record in records])
    harm_probabilities = np.asarray([float(record["harm_probability"]) for record in records])
    for record in records:
        record[f"{policy}_gain_over_fixed"] = float(record[selected_key]) - float(
            record["fixed_score"]
        )
    action_qids = {
        action: len(
            {str(record["qid"]) for record in records if record[action_key] == action}
        )
        for action in ("keep", "restart")
    }
    return {
        "selected_mean_f1": float(np.mean(selected)),
        "fixed_mean_f1": float(np.mean(fixed)),
        "oracle_mean_f1": float(np.mean(oracle)),
        "selected_minus_fixed": _cluster_bootstrap(
            records, f"{policy}_gain_over_fixed", samples=bootstrap_samples
        ),
        "normalized_oracle_recovery": (
            float(np.mean(selected) - np.mean(fixed)) / denominator if denominator > 0 else 0.0
        ),
        "intervention_coverage": float(
            np.mean([record[action_key] == "restart" for record in records])
        ),
        "selected_action_qids": action_qids,
        "selected_harm_rows": int(harms.sum()),
        "selected_rescue_rows": int(rescues.sum()),
        "harm_prevalence": float(np.mean(harm_labels)),
        "harm_auprc": float(average_precision_score(harm_labels, harm_probabilities)),
        "harm_auroc": (
            float(roc_auc_score(harm_labels, harm_probabilities))
            if len(set(harm_labels.tolist())) > 1
            else None
        ),
        "harm_brier": float(brier_score_loss(harm_labels, harm_probabilities)),
    }


def run_probe(
    rows: Sequence[Dict[str, Any]],
    variants: Sequence[str] = ("numeric", "no_prefix_tfidf", "full_tfidf"),
    outer_folds: int = 5,
    inner_folds: int = 4,
    bootstrap_samples: int = 10_000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import numpy as np
    import scipy
    import sklearn

    all_records: List[Dict[str, Any]] = []
    dimensions: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for variant in variants:
        for fold, (train_indices, test_indices) in enumerate(
            _group_splits(rows, outer_folds)
        ):
            train_rows = [rows[int(index)] for index in train_indices]
            test_rows = [rows[int(index)] for index in test_indices]
            inner_delta, inner_harm = _inner_predictions(train_rows, variant, inner_folds)
            value_policy = _choose_policy(
                train_rows, inner_delta, inner_harm, risk_aware=False
            )
            risk_policy = _choose_policy(
                train_rows, inner_delta, inner_harm, risk_aware=True
            )
            delta_prediction, harm_prediction, dims = _fit_predict(
                train_rows, test_rows, variant
            )
            dimensions[variant].append(dims)
            keep_train = float(np.mean([row["labels"]["keep_f1"] for row in train_rows]))
            restart_train = float(
                np.mean([row["labels"]["restart_f1"] for row in train_rows])
            )
            best_fixed = "restart" if restart_train >= keep_train else "keep"
            for row, delta_pred, harm_prob in zip(
                test_rows, delta_prediction, harm_prediction
            ):
                record: Dict[str, Any] = {
                    "probe_version": PROBE_VERSION,
                    "variant": variant,
                    "outer_fold": fold,
                    "sample_index": int(row["sample_index"]),
                    "qid": str(row["qid"]),
                    "state_id": str(row["state_id"]),
                    "query_source": str(row["query_source"]),
                    "delta_prediction": float(delta_pred),
                    "harm_probability": float(harm_prob),
                    "keep_f1": float(row["labels"]["keep_f1"]),
                    "restart_f1": float(row["labels"]["restart_f1"]),
                    "restart_harm": int(row["labels"]["restart_harm"]),
                    "restart_rescue": int(row["labels"]["restart_rescue"]),
                    "fold_best_fixed_action": best_fixed,
                    "fixed_score": float(row["labels"][f"{best_fixed}_f1"]),
                    "oracle_score": max(
                        float(row["labels"]["keep_f1"]),
                        float(row["labels"]["restart_f1"]),
                    ),
                    "value_lambda_harm": value_policy["lambda_harm"],
                    "value_threshold": value_policy["threshold"],
                    "risk_lambda_harm": risk_policy["lambda_harm"],
                    "risk_threshold": risk_policy["threshold"],
                }
                for policy_name, policy in (
                    ("value", value_policy),
                    ("risk", risk_policy),
                ):
                    adjusted = float(delta_pred) - float(policy["lambda_harm"]) * float(
                        harm_prob
                    )
                    action = "restart" if adjusted > float(policy["threshold"]) else "keep"
                    record[f"{policy_name}_selected_action"] = action
                    record[f"{policy_name}_selected_score"] = float(
                        row["labels"][f"{action}_f1"]
                    )
                all_records.append(record)

    variant_reports: Dict[str, Any] = {}
    for variant in variants:
        variant_records = [record for record in all_records if record["variant"] == variant]
        variant_reports[variant] = {
            "value_policy": _aggregate(
                variant_records, "value", bootstrap_samples=bootstrap_samples
            ),
            "risk_policy": _aggregate(
                variant_records, "risk", bootstrap_samples=bootstrap_samples
            ),
            "feature_dimensions_by_fold": dimensions[variant],
        }
    full_recovery = variant_reports.get("full_tfidf", {}).get("risk_policy", {}).get(
        "normalized_oracle_recovery"
    )
    no_prefix_recovery = variant_reports.get("no_prefix_tfidf", {}).get(
        "risk_policy", {}
    ).get("normalized_oracle_recovery")
    report = {
        "probe_version": PROBE_VERSION,
        "rows": len(rows),
        "qids": len({str(row["qid"]) for row in rows}),
        "states": len({str(row["state_id"]) for row in rows}),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "variants": variant_reports,
        "full_minus_no_prefix_recovery": (
            float(full_recovery) - float(no_prefix_recovery)
            if full_recovery is not None and no_prefix_recovery is not None
            else None
        ),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "全部预测为 qid-group nested-CV 的 out-of-fold 结果；"
            "外折测试结果不参与阈值、harm 权重或最佳固定动作选择。"
        ),
    }
    return report, all_records


def _write_json_exclusive(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=["numeric", "no_prefix_tfidf", "full_tfidf"],
        default=None,
    )
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof_output", required=True)
    args = parser.parse_args()
    variants = args.variant or ["numeric", "no_prefix_tfidf", "full_tfidf"]
    report, records = run_probe(
        load_rows(args.supervision),
        variants=variants,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    _write_json_exclusive(args.output, report)
    _write_jsonl_exclusive(args.oof_output, records)
    compact = {
        "probe_version": report["probe_version"],
        "rows": report["rows"],
        "qids": report["qids"],
        "full_minus_no_prefix_recovery": report["full_minus_no_prefix_recovery"],
        "variants": {
            name: {
                policy: {
                    key: values[policy][key]
                    for key in (
                        "selected_mean_f1",
                        "fixed_mean_f1",
                        "oracle_mean_f1",
                        "normalized_oracle_recovery",
                        "selected_harm_rows",
                        "selected_rescue_rows",
                        "harm_auprc",
                    )
                }
                for policy in ("value_policy", "risk_policy")
            }
            for name, values in report["variants"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
