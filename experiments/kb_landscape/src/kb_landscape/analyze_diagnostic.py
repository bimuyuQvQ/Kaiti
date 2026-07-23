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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="留一知识库分析局部景观的增量预测价值")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def _selected_rewards(predictions: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choices = np.argmax(predictions, axis=1)
    selected = rewards[np.arange(len(rewards)), choices]
    return selected, choices


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, len(differences), size=len(differences))
        means[sample_index] = differences[indices].mean()
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _strategy_record(
    *,
    corpus: str,
    strategy: str,
    selected: np.ndarray,
    choices: np.ndarray,
    keep: np.ndarray,
    oracle: np.ndarray,
    oracle_choices: np.ndarray,
) -> dict[str, float | str]:
    return {
        "held_out_corpus": corpus,
        "strategy": strategy,
        "queries": int(len(selected)),
        "mean_ndcg": float(selected.mean()),
        "oracle_regret": float((oracle - selected).mean()),
        "harm_rate_vs_keep": float((selected < keep - 1e-12).mean()),
        "win_rate_vs_keep": float((selected > keep + 1e-12).mean()),
        "oracle_action_accuracy": float((choices == oracle_choices).mean()),
    }


def analyze(frame: pd.DataFrame, *, seed: int, bootstrap_samples: int) -> tuple[pd.DataFrame, dict]:
    action_columns = sorted(column for column in frame.columns if column.startswith("ndcg__"))
    actions = [column.removeprefix("ndcg__") for column in action_columns]
    feature_columns = sorted(column for column in frame.columns if column.startswith("feat_"))
    if "keep" not in actions:
        raise ValueError("输入中缺少 ndcg__keep")
    corpora = sorted(frame["corpus"].unique())
    if len(corpora) < 2:
        raise ValueError("留一知识库分析至少需要两个 corpus")

    records: list[dict] = []
    per_query_outputs: list[pd.DataFrame] = []
    for fold_index, held_out in enumerate(corpora):
        train = frame[frame["corpus"] != held_out].reset_index(drop=True)
        test = frame[frame["corpus"] == held_out].reset_index(drop=True)
        y_train = train[action_columns].to_numpy(dtype=np.float64)
        y_test = test[action_columns].to_numpy(dtype=np.float64)
        keep = test["ndcg__keep"].to_numpy(dtype=np.float64)
        oracle_choices = np.argmax(y_test, axis=1)
        oracle = y_test[np.arange(len(y_test)), oracle_choices]

        strategy_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        keep_choice = actions.index("keep")
        strategy_values["keep"] = (keep, np.full(len(test), keep_choice, dtype=np.int64))

        global_choice = int(np.argmax(y_train.mean(axis=0)))
        strategy_values["global_best"] = (
            y_test[:, global_choice],
            np.full(len(test), global_choice, dtype=np.int64),
        )

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
        strategy_values["query_only"] = _selected_rewards(query_model.predict(query_test), y_test)

        landscape_model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=seed + fold_index,
        )
        landscape_model.fit(train[feature_columns].to_numpy(dtype=np.float64), y_train)
        strategy_values["landscape_only"] = _selected_rewards(
            landscape_model.predict(test[feature_columns].to_numpy(dtype=np.float64)),
            y_test,
        )

        scaler = StandardScaler()
        landscape_train = scaler.fit_transform(train[feature_columns].to_numpy(dtype=np.float64))
        landscape_test = scaler.transform(test[feature_columns].to_numpy(dtype=np.float64))
        combined_train = sp.hstack([query_train, sp.csr_matrix(landscape_train)], format="csr")
        combined_test = sp.hstack([query_test, sp.csr_matrix(landscape_test)], format="csr")
        combined_model = Ridge(alpha=5.0)
        combined_model.fit(combined_train, y_train)
        strategy_values["query_landscape"] = _selected_rewards(combined_model.predict(combined_test), y_test)
        strategy_values["oracle"] = (oracle, oracle_choices)

        fold_output = test[["corpus", "query_id", "query"]].copy()
        fold_output["oracle_ndcg"] = oracle
        fold_output["oracle_action"] = [actions[index] for index in oracle_choices]
        for strategy, (selected, choices) in strategy_values.items():
            records.append(
                _strategy_record(
                    corpus=held_out,
                    strategy=strategy,
                    selected=selected,
                    choices=choices,
                    keep=keep,
                    oracle=oracle,
                    oracle_choices=oracle_choices,
                )
            )
            fold_output[f"selected_ndcg__{strategy}"] = selected
            fold_output[f"selected_action__{strategy}"] = [actions[index] for index in choices]
        per_query_outputs.append(fold_output)

    result = pd.DataFrame(records)
    per_query = pd.concat(per_query_outputs, ignore_index=True)
    comparisons: dict[str, dict[str, float]] = {}
    for baseline in ("global_best", "query_only", "keep"):
        comparisons[f"landscape_only_vs_{baseline}"] = _bootstrap_difference(
            per_query["selected_ndcg__landscape_only"].to_numpy(),
            per_query[f"selected_ndcg__{baseline}"].to_numpy(),
            samples=bootstrap_samples,
            seed=seed,
        )
        comparisons[f"query_landscape_vs_{baseline}"] = _bootstrap_difference(
            per_query["selected_ndcg__query_landscape"].to_numpy(),
            per_query[f"selected_ndcg__{baseline}"].to_numpy(),
            samples=bootstrap_samples,
            seed=seed + 1,
        )
    summary = {
        "corpora": corpora,
        "queries": int(len(frame)),
        "actions": actions,
        "features": feature_columns,
        "macro_mean_ndcg": result.groupby("strategy")["mean_ndcg"].mean().to_dict(),
        "macro_oracle_regret": result.groupby("strategy")["oracle_regret"].mean().to_dict(),
        "comparisons": comparisons,
    }
    per_query.attrs["summary"] = summary
    return result, {"summary": summary, "per_query": per_query}


def main() -> None:
    args = _parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    frame = pd.concat(frames, ignore_index=True)
    result, details = analyze(frame, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "analysis.csv", index=False)
    details["per_query"].to_csv(output_dir / "per_query_predictions.csv", index=False)
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(details["summary"], handle, ensure_ascii=False, indent=2)
    print(json.dumps(details["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
