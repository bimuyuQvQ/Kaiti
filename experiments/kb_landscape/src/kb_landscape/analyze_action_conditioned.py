from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from kb_landscape.analyze_diagnostic import _bootstrap_difference


STRATEGIES = (
    "keep",
    "global_best",
    "query_action",
    "action_landscape",
    "query_action_landscape",
    "oracle",
)
CALIBRATED_STRATEGIES = (
    "query_action_calibrated",
    "action_landscape_calibrated",
    "query_action_landscape_calibrated",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用动作条件化检索景观预测查询动作效用")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("loco", "within"), default="loco")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def _action_feature_names(frame: pd.DataFrame, actions: list[str]) -> list[str]:
    feature_sets = []
    for action in actions:
        prefix = f"action_feat__{action}__"
        feature_sets.append(
            {column.removeprefix(prefix) for column in frame.columns if column.startswith(prefix)}
        )
    common = set.intersection(*feature_sets) if feature_sets else set()
    if not common:
        raise ValueError("输入中缺少共同的 action_feat__<action>__ 特征")
    return sorted(common)


def _make_long(
    frame: pd.DataFrame,
    *,
    actions: list[str],
    action_features: list[str],
    static_features: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for action_index, action in enumerate(actions):
        action_frame = frame[["_row_id", "corpus", "query_id"]].copy()
        action_frame["action"] = action
        action_frame["action_index"] = action_index
        action_frame["candidate_query"] = frame[f"query__{action}"].fillna("")
        action_frame["reward"] = frame[f"ndcg__{action}"].astype(float)
        action_frame["gain"] = action_frame["reward"] - frame["ndcg__keep"].astype(float)
        for feature_name in action_features:
            action_frame[f"abs__{feature_name}"] = frame[
                f"action_feat__{action}__{feature_name}"
            ].astype(float)
            keep_values = frame[f"action_feat__keep__{feature_name}"].astype(float)
            action_frame[f"delta__{feature_name}"] = (
                action_frame[f"abs__{feature_name}"] - keep_values
            )
        for feature_name in static_features:
            action_frame[f"static__{feature_name.removeprefix('feat_')}"] = frame[
                feature_name
            ].astype(float)
        rows.append(action_frame)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_columns: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    action_count = int(train["action_index"].max()) + 1
    action_train = np.eye(action_count, dtype=np.float64)[train["action_index"].to_numpy()]
    action_test = np.eye(action_count, dtype=np.float64)[test["action_index"].to_numpy()]
    numeric_train = np.hstack([train[numeric_columns].to_numpy(dtype=np.float64), action_train])
    numeric_test = np.hstack([test[numeric_columns].to_numpy(dtype=np.float64), action_test])
    target = train["gain"].to_numpy(dtype=np.float64)

    query_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    query_train = query_vectorizer.fit_transform(train["candidate_query"].fillna(""))
    query_test = query_vectorizer.transform(test["candidate_query"].fillna(""))
    query_model = Ridge(alpha=5.0)
    query_model.fit(sp.hstack([query_train, sp.csr_matrix(action_train)]), target)
    query_predictions = query_model.predict(
        sp.hstack([query_test, sp.csr_matrix(action_test)])
    )

    landscape_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=8,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )
    landscape_model.fit(numeric_train, target)
    landscape_predictions = landscape_model.predict(numeric_test)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(numeric_train)
    scaled_test = scaler.transform(numeric_test)
    combined_train = sp.hstack([query_train, sp.csr_matrix(scaled_train)], format="csr")
    combined_test = sp.hstack([query_test, sp.csr_matrix(scaled_test)], format="csr")
    combined_model = Ridge(alpha=5.0)
    combined_model.fit(combined_train, target)
    combined_predictions = combined_model.predict(combined_test)
    return {
        "query_action": query_predictions,
        "action_landscape": landscape_predictions,
        "query_action_landscape": combined_predictions,
    }


def _folds(
    frame: pd.DataFrame,
    *,
    protocol: str,
    folds: int,
    seed: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    if protocol == "loco":
        for corpus in sorted(frame["corpus"].unique()):
            test = np.flatnonzero(frame["corpus"].to_numpy() == corpus)
            train = np.flatnonzero(frame["corpus"].to_numpy() != corpus)
            splits.append((corpus, train, test))
        return splits

    for corpus_index, corpus in enumerate(sorted(frame["corpus"].unique())):
        corpus_indices = np.flatnonzero(frame["corpus"].to_numpy() == corpus)
        splitter = KFold(
            n_splits=min(folds, len(corpus_indices)),
            shuffle=True,
            random_state=seed + corpus_index * 1000,
        )
        for fold_index, (train_local, test_local) in enumerate(splitter.split(corpus_indices)):
            splits.append(
                (
                    f"{corpus}/fold-{fold_index}",
                    corpus_indices[train_local],
                    corpus_indices[test_local],
                )
            )
    return splits


def _to_wide_predictions(
    long_frame: pd.DataFrame,
    values: np.ndarray,
    *,
    row_ids: np.ndarray,
    action_count: int,
) -> np.ndarray:
    positions = {
        (int(row_id), int(action_index)): position
        for position, (row_id, action_index) in enumerate(
            zip(long_frame["_row_id"], long_frame["action_index"])
        )
    }
    output = np.empty((len(row_ids), action_count), dtype=np.float64)
    for local_index, row_id in enumerate(row_ids):
        for action_index in range(action_count):
            output[local_index, action_index] = values[positions[(int(row_id), action_index)]]
    return output


def _tune_threshold(
    predictions: np.ndarray,
    rewards: np.ndarray,
    *,
    keep_index: int,
) -> tuple[float, float]:
    alternative_indices = [index for index in range(predictions.shape[1]) if index != keep_index]
    best_alternative = predictions[:, alternative_indices].max(axis=1)
    positive = best_alternative[best_alternative > 0]
    candidates = {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1}
    if len(positive):
        candidates.update(float(value) for value in np.quantile(
            positive,
            [0.25, 0.5, 0.75, 0.9, 0.95, 0.975],
        ))

    best_threshold = 0.0
    best_reward = -np.inf
    for threshold in sorted(candidates):
        choices = np.full(len(predictions), keep_index, dtype=np.int64)
        alternative_local = np.argmax(predictions[:, alternative_indices], axis=1)
        use_alternative = best_alternative > threshold
        choices[use_alternative] = np.asarray(alternative_indices)[
            alternative_local[use_alternative]
        ]
        selected = rewards[np.arange(len(rewards)), choices]
        mean_reward = float(selected.mean())
        if mean_reward > best_reward + 1e-12 or (
            abs(mean_reward - best_reward) <= 1e-12 and threshold > best_threshold
        ):
            best_reward = mean_reward
            best_threshold = float(threshold)
    return best_threshold, best_reward


def _apply_threshold(
    predictions: np.ndarray,
    *,
    keep_index: int,
    threshold: float,
) -> np.ndarray:
    adjusted = predictions.copy()
    alternative_indices = [index for index in range(predictions.shape[1]) if index != keep_index]
    adjusted[:, alternative_indices] -= threshold + 1e-12
    adjusted[:, keep_index] = 0.0
    return adjusted


def analyze(
    frame: pd.DataFrame,
    *,
    protocol: str,
    folds: int,
    seed: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = frame.reset_index(drop=True).copy()
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    action_columns = sorted(column for column in frame.columns if column.startswith("ndcg__"))
    actions = [column.removeprefix("ndcg__") for column in action_columns]
    if "keep" not in actions:
        raise ValueError("输入中缺少 ndcg__keep")
    action_features = _action_feature_names(frame, actions)
    static_features = sorted(
        column
        for column in frame.columns
        if column.startswith("feat_") and not column.startswith("feat_probe_")
    )
    long_frame = _make_long(
        frame,
        actions=actions,
        action_features=action_features,
        static_features=static_features,
    )
    numeric_columns = sorted(
        column
        for column in long_frame.columns
        if column.startswith(("abs__", "delta__", "static__"))
    )

    strategies = list(STRATEGIES)
    if protocol == "loco":
        strategies.extend(CALIBRATED_STRATEGIES)
    prediction_columns = {
        strategy: np.full((len(frame), len(actions)), np.nan, dtype=np.float64)
        for strategy in strategies
    }
    rewards = frame[action_columns].to_numpy(dtype=np.float64)
    keep_index = actions.index("keep")
    prediction_columns["keep"][:, :] = -np.inf
    prediction_columns["keep"][:, keep_index] = 0.0
    prediction_columns["oracle"] = rewards.copy()

    fold_labels = np.full(len(frame), "", dtype=object)
    threshold_records: list[dict[str, float | str]] = []
    for fold_index, (fold_label, train_indices, test_indices) in enumerate(
        _folds(frame, protocol=protocol, folds=folds, seed=seed)
    ):
        fold_labels[test_indices] = fold_label
        train_ids = set(frame.iloc[train_indices]["_row_id"].astype(int))
        test_ids = set(frame.iloc[test_indices]["_row_id"].astype(int))
        train_long = long_frame[long_frame["_row_id"].astype(int).isin(train_ids)].copy()
        test_long = long_frame[long_frame["_row_id"].astype(int).isin(test_ids)].copy()

        predicted = _fit_predict(
            train_long,
            test_long,
            numeric_columns=numeric_columns,
            seed=seed + fold_index,
        )
        train_means = frame.iloc[train_indices][action_columns].mean(axis=0).to_numpy()
        global_choice = int(np.argmax(train_means))
        prediction_columns["global_best"][test_indices, :] = -np.inf
        prediction_columns["global_best"][test_indices, global_choice] = 0.0

        for strategy, values in predicted.items():
            prediction_columns[strategy][test_indices] = _to_wide_predictions(
                test_long,
                values,
                row_ids=frame.iloc[test_indices]["_row_id"].to_numpy(dtype=np.int64),
                action_count=len(actions),
            )
            prediction_columns[strategy][test_indices, keep_index] = 0.0

        if protocol == "loco":
            inner_predictions = {
                strategy: np.full((len(train_indices), len(actions)), np.nan, dtype=np.float64)
                for strategy in predicted
            }
            train_positions = {
                int(row_id): position
                for position, row_id in enumerate(
                    frame.iloc[train_indices]["_row_id"].to_numpy(dtype=np.int64)
                )
            }
            train_corpora = frame.iloc[train_indices]["corpus"].to_numpy()
            for inner_index, validation_corpus in enumerate(sorted(set(train_corpora))):
                validation_mask = train_corpora == validation_corpus
                inner_validation_indices = train_indices[validation_mask]
                inner_train_indices = train_indices[~validation_mask]
                inner_train_ids = set(frame.iloc[inner_train_indices]["_row_id"].astype(int))
                inner_validation_ids = set(
                    frame.iloc[inner_validation_indices]["_row_id"].astype(int)
                )
                inner_train_long = long_frame[
                    long_frame["_row_id"].astype(int).isin(inner_train_ids)
                ].copy()
                inner_validation_long = long_frame[
                    long_frame["_row_id"].astype(int).isin(inner_validation_ids)
                ].copy()
                inner_predicted = _fit_predict(
                    inner_train_long,
                    inner_validation_long,
                    numeric_columns=numeric_columns,
                    seed=seed + fold_index * 100 + inner_index,
                )
                inner_wide_positions = np.asarray(
                    [train_positions[int(row_id)] for row_id in inner_validation_indices],
                    dtype=np.int64,
                )
                for strategy, values in inner_predicted.items():
                    inner_predictions[strategy][inner_wide_positions] = _to_wide_predictions(
                        inner_validation_long,
                        values,
                        row_ids=frame.iloc[inner_validation_indices]["_row_id"].to_numpy(
                            dtype=np.int64
                        ),
                        action_count=len(actions),
                    )
                    inner_predictions[strategy][inner_wide_positions, keep_index] = 0.0

            train_rewards = rewards[train_indices]
            for strategy, values in inner_predictions.items():
                if np.isnan(values).any():
                    raise RuntimeError(f"{fold_label} 的训练内校准预测不完整：{strategy}")
                threshold, calibration_reward = _tune_threshold(
                    values,
                    train_rewards,
                    keep_index=keep_index,
                )
                calibrated_strategy = f"{strategy}_calibrated"
                prediction_columns[calibrated_strategy][test_indices] = _apply_threshold(
                    prediction_columns[strategy][test_indices],
                    keep_index=keep_index,
                    threshold=threshold,
                )
                threshold_records.append(
                    {
                        "fold": fold_label,
                        "strategy": calibrated_strategy,
                        "threshold": threshold,
                        "inner_validation_mean_ndcg": calibration_reward,
                    }
                )

    if any(np.isnan(values).any() for values in prediction_columns.values()):
        raise RuntimeError("存在未填充的折外预测")

    output = frame[["corpus", "query_id", "query"]].copy()
    output["fold"] = fold_labels
    oracle = rewards.max(axis=1)
    keep = rewards[:, keep_index]
    output["oracle_ndcg"] = oracle
    records: list[dict] = []
    for strategy, predicted in prediction_columns.items():
        choices = np.argmax(predicted, axis=1)
        selected = rewards[np.arange(len(rewards)), choices]
        output[f"selected_ndcg__{strategy}"] = selected
        output[f"selected_action__{strategy}"] = [actions[index] for index in choices]
        for corpus in sorted(frame["corpus"].unique()):
            mask = frame["corpus"].to_numpy() == corpus
            records.append(
                {
                    "corpus": corpus,
                    "strategy": strategy,
                    "queries": int(mask.sum()),
                    "mean_ndcg": float(selected[mask].mean()),
                    "oracle_regret": float((oracle[mask] - selected[mask]).mean()),
                    "harm_rate_vs_keep": float((selected[mask] < keep[mask] - 1e-12).mean()),
                    "win_rate_vs_keep": float((selected[mask] > keep[mask] + 1e-12).mean()),
                    "oracle_utility_accuracy": float(
                        (selected[mask] >= oracle[mask] - 1e-12).mean()
                    ),
                }
            )

    result = pd.DataFrame(records)
    comparisons: dict[str, dict[str, float]] = {}
    compared_strategies = ["action_landscape", "query_action_landscape"]
    if protocol == "loco":
        compared_strategies.extend(
            ["action_landscape_calibrated", "query_action_landscape_calibrated"]
        )
    for baseline in ("global_best", "query_action", "keep"):
        for offset, strategy in enumerate(compared_strategies):
            comparisons[f"{strategy}_vs_{baseline}"] = _bootstrap_difference(
                output[f"selected_ndcg__{strategy}"].to_numpy(),
                output[f"selected_ndcg__{baseline}"].to_numpy(),
                samples=bootstrap_samples,
                seed=seed + offset,
            )
    summary = {
        "protocol": protocol,
        "corpora": sorted(frame["corpus"].unique()),
        "queries": int(len(frame)),
        "actions": actions,
        "action_features": action_features,
        "numeric_feature_count": len(numeric_columns) + len(actions),
        "calibration_thresholds": threshold_records,
        "macro_mean_ndcg": result.groupby("strategy")["mean_ndcg"].mean().to_dict(),
        "macro_oracle_regret": result.groupby("strategy")["oracle_regret"].mean().to_dict(),
        "comparisons": comparisons,
    }
    return result, output, summary


def main() -> None:
    args = _parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    result, per_query, summary = analyze(
        frame,
        protocol=args.protocol,
        folds=args.folds,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "analysis.csv", index=False)
    per_query.to_csv(output_dir / "per_query_predictions.csv", index=False)
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
