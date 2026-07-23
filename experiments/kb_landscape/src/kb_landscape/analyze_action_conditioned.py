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

    prediction_columns = {
        strategy: np.full((len(frame), len(actions)), np.nan, dtype=np.float64)
        for strategy in STRATEGIES
    }
    rewards = frame[action_columns].to_numpy(dtype=np.float64)
    keep_index = actions.index("keep")
    prediction_columns["keep"][:, :] = -np.inf
    prediction_columns["keep"][:, keep_index] = 0.0
    prediction_columns["oracle"] = rewards.copy()

    fold_labels = np.full(len(frame), "", dtype=object)
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

        test_order = {
            (int(row_id), int(action_index)): position
            for position, (row_id, action_index) in enumerate(
                zip(test_long["_row_id"], test_long["action_index"])
            )
        }
        for strategy, values in predicted.items():
            for wide_index in test_indices:
                row_id = int(frame.iloc[wide_index]["_row_id"])
                for action_index in range(len(actions)):
                    prediction_columns[strategy][wide_index, action_index] = values[
                        test_order[(row_id, action_index)]
                    ]
            prediction_columns[strategy][test_indices, keep_index] = 0.0

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
    for baseline in ("global_best", "query_action", "keep"):
        for offset, strategy in enumerate(("action_landscape", "query_action_landscape")):
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
