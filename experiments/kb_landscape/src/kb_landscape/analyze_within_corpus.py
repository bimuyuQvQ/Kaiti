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

from kb_landscape.analyze_diagnostic import _bootstrap_difference, _selected_rewards


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="知识库内交叉验证局部检索景观的动作预测能力")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def _evaluate_corpus(
    frame: pd.DataFrame,
    *,
    actions: list[str],
    action_columns: list[str],
    feature_columns: list[str],
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if frame["query_id"].duplicated().any():
        raise ValueError(f"{frame['corpus'].iloc[0]} 中存在重复 query_id")
    if len(frame) < 2:
        raise ValueError("知识库内交叉验证至少需要两个查询")

    split_count = min(folds, len(frame))
    if split_count < 2:
        raise ValueError("folds 至少为 2")

    frame = frame.reset_index(drop=True)
    rewards = frame[action_columns].to_numpy(dtype=np.float64)
    keep_choice = actions.index("keep")
    oracle_choices = np.argmax(rewards, axis=1)
    oracle = rewards[np.arange(len(rewards)), oracle_choices]
    predictions: dict[str, np.ndarray] = {
        "keep": np.full((len(frame), len(actions)), -np.inf, dtype=np.float64),
        "global_best": np.full((len(frame), len(actions)), -np.inf, dtype=np.float64),
        "query_only": np.zeros_like(rewards),
        "landscape_only": np.zeros_like(rewards),
        "query_landscape": np.zeros_like(rewards),
        "oracle": rewards.copy(),
    }
    predictions["keep"][:, keep_choice] = 0.0

    splitter = KFold(n_splits=split_count, shuffle=True, random_state=seed)
    fold_ids = np.full(len(frame), -1, dtype=np.int64)
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(frame)):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        y_train = rewards[train_indices]
        fold_ids[test_indices] = fold_index

        global_choice = int(np.argmax(y_train.mean(axis=0)))
        predictions["global_best"][test_indices, global_choice] = 0.0

        query_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )
        query_train = query_vectorizer.fit_transform(train["query"].fillna(""))
        query_test = query_vectorizer.transform(test["query"].fillna(""))
        query_model = Ridge(alpha=5.0)
        query_model.fit(query_train, y_train)
        predictions["query_only"][test_indices] = query_model.predict(query_test)

        landscape_model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=seed + fold_index,
        )
        landscape_train_raw = train[feature_columns].to_numpy(dtype=np.float64)
        landscape_test_raw = test[feature_columns].to_numpy(dtype=np.float64)
        landscape_model.fit(landscape_train_raw, y_train)
        predictions["landscape_only"][test_indices] = landscape_model.predict(landscape_test_raw)

        scaler = StandardScaler()
        landscape_train = scaler.fit_transform(landscape_train_raw)
        landscape_test = scaler.transform(landscape_test_raw)
        combined_train = sp.hstack([query_train, sp.csr_matrix(landscape_train)], format="csr")
        combined_test = sp.hstack([query_test, sp.csr_matrix(landscape_test)], format="csr")
        combined_model = Ridge(alpha=5.0)
        combined_model.fit(combined_train, y_train)
        predictions["query_landscape"][test_indices] = combined_model.predict(combined_test)

    output = frame[["corpus", "query_id", "query"]].copy()
    output["fold"] = fold_ids
    output["oracle_ndcg"] = oracle
    output["oracle_action"] = [actions[index] for index in oracle_choices]
    keep = rewards[:, keep_choice]
    records: list[dict] = []
    for strategy, predicted_rewards in predictions.items():
        selected, choices = _selected_rewards(predicted_rewards, rewards)
        output[f"selected_ndcg__{strategy}"] = selected
        output[f"selected_action__{strategy}"] = [actions[index] for index in choices]
        records.append(
            {
                "corpus": frame["corpus"].iloc[0],
                "strategy": strategy,
                "queries": int(len(frame)),
                "folds": int(split_count),
                "mean_ndcg": float(selected.mean()),
                "oracle_regret": float((oracle - selected).mean()),
                "harm_rate_vs_keep": float((selected < keep - 1e-12).mean()),
                "win_rate_vs_keep": float((selected > keep + 1e-12).mean()),
                "oracle_action_accuracy_strict": float((choices == oracle_choices).mean()),
                "oracle_utility_accuracy": float((selected >= oracle - 1e-12).mean()),
            }
        )
    return output, records


def analyze(
    frame: pd.DataFrame,
    *,
    folds: int,
    seed: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    action_columns = sorted(column for column in frame.columns if column.startswith("ndcg__"))
    actions = [column.removeprefix("ndcg__") for column in action_columns]
    feature_columns = sorted(column for column in frame.columns if column.startswith("feat_"))
    if "keep" not in actions:
        raise ValueError("输入中缺少 ndcg__keep")
    if not feature_columns:
        raise ValueError("输入中缺少 feat_ 特征")

    outputs: list[pd.DataFrame] = []
    records: list[dict] = []
    for corpus_index, corpus in enumerate(sorted(frame["corpus"].unique())):
        corpus_output, corpus_records = _evaluate_corpus(
            frame[frame["corpus"] == corpus],
            actions=actions,
            action_columns=action_columns,
            feature_columns=feature_columns,
            folds=folds,
            seed=seed + corpus_index * 1000,
        )
        outputs.append(corpus_output)
        records.extend(corpus_records)

    per_query = pd.concat(outputs, ignore_index=True)
    result = pd.DataFrame(records)
    comparisons: dict[str, dict[str, float]] = {}
    for baseline in ("global_best", "query_only", "keep"):
        for strategy_offset, strategy in enumerate(("landscape_only", "query_landscape")):
            comparisons[f"{strategy}_vs_{baseline}"] = _bootstrap_difference(
                per_query[f"selected_ndcg__{strategy}"].to_numpy(),
                per_query[f"selected_ndcg__{baseline}"].to_numpy(),
                samples=bootstrap_samples,
                seed=seed + strategy_offset,
            )
    summary = {
        "protocol": f"{folds}-fold within-corpus cross-validation",
        "corpora": sorted(frame["corpus"].unique()),
        "queries": int(len(frame)),
        "actions": actions,
        "features": feature_columns,
        "macro_mean_ndcg": result.groupby("strategy")["mean_ndcg"].mean().to_dict(),
        "macro_oracle_regret": result.groupby("strategy")["oracle_regret"].mean().to_dict(),
        "comparisons": comparisons,
    }
    return result, per_query, summary


def main() -> None:
    args = _parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    result, per_query, summary = analyze(
        frame,
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
