from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .actions import DEFAULT_ACTIONS, generate_candidates
from .bm25 import BM25Index
from .features import probe_stability_features, raw_landscape_features
from .io import load_beir_dataset, load_external_candidates
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行知识库局部检索景观诊断")
    parser.add_argument("--dataset", required=True, help="BEIR 格式数据集目录")
    parser.add_argument("--corpus-name", required=True, help="输出中记录的知识库名称")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--external-candidates", help="可选的外部查询候选 JSONL")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=list(DEFAULT_ACTIONS),
        choices=list(DEFAULT_ACTIONS),
        help="启用的内置查询操作；外部候选不受此参数影响",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    dataset = load_beir_dataset(args.dataset, split=args.split)
    external = load_external_candidates(args.external_candidates)
    queries = list(dataset.queries)
    if args.max_queries is not None and len(queries) > args.max_queries:
        random.Random(args.seed).shuffle(queries)
        queries = sorted(queries[: args.max_queries], key=lambda item: item.query_id)

    texts = [document.full_text for document in dataset.documents]
    index = BM25Index().fit(texts)
    corpus_tokens = [index.tokenize(text) for text in texts]
    doc_ids = np.asarray([document.doc_id for document in dataset.documents], dtype=object)

    rows: list[dict] = []
    for query_number, query in enumerate(queries, start=1):
        raw_result = index.search(query.text, top_k=args.top_k)
        candidates = generate_candidates(
            query.text,
            index,
            raw_result,
            corpus_tokens,
            external=external.get(query.query_id),
            actions=tuple(args.actions),
        )
        results = {action: index.search(text, top_k=args.top_k) for action, text in candidates.items()}
        qrels = dataset.qrels[query.query_id]
        row: dict[str, object] = {
            "corpus": args.corpus_name,
            "query_id": query.query_id,
            "query": query.text,
        }
        row.update(raw_landscape_features(query.text, index, raw_result, corpus_tokens))
        row.update(probe_stability_features(results))

        action_scores: dict[str, float] = {}
        for action, result in results.items():
            retrieved = [str(value) for value in doc_ids[result.indices]]
            ndcg = ndcg_at_k(retrieved, qrels, k=args.top_k)
            action_scores[action] = ndcg
            row[f"query__{action}"] = candidates[action]
            row[f"ndcg__{action}"] = ndcg
            row[f"recall__{action}"] = recall_at_k(retrieved, qrels, k=args.top_k)
            row[f"mrr__{action}"] = mrr_at_k(retrieved, qrels, k=args.top_k)
            row[f"retrieved__{action}"] = json.dumps(retrieved, ensure_ascii=False)

        best_score = max(action_scores.values())
        tied = [action for action, score in action_scores.items() if abs(score - best_score) <= 1e-12]
        best_action = "keep" if "keep" in tied else tied[0]
        sorted_scores = sorted(action_scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        keep_score = action_scores["keep"]
        row.update(
            {
                "best_action": best_action,
                "best_ndcg": best_score,
                "keep_ndcg": keep_score,
                "oracle_gain": best_score - keep_score,
                "best_margin": best_score - second_score,
                "tie_count": len(tied),
            }
        )
        rows.append(row)
        if query_number % 25 == 0 or query_number == len(queries):
            print(f"[{args.corpus_name}] {query_number}/{len(queries)}", flush=True)

    frame = pd.DataFrame(rows)
    action_columns = sorted(column for column in frame.columns if column.startswith("ndcg__"))
    summary = {
        "corpus": args.corpus_name,
        "dataset": str(Path(args.dataset).resolve()),
        "documents": len(dataset.documents),
        "queries": len(frame),
        "vocabulary": len(index.vocabulary),
        "top_k": args.top_k,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "mean_keep_ndcg": float(frame["keep_ndcg"].mean()),
        "mean_oracle_ndcg": float(frame["best_ndcg"].mean()),
        "mean_oracle_gain": float(frame["oracle_gain"].mean()),
        "oracle_action_counts": dict(Counter(frame["best_action"])),
        "tie_rate": float((frame["tie_count"] > 1).mean()),
        "action_mean_ndcg": {
            column.removeprefix("ndcg__"): float(frame[column].mean()) for column in action_columns
        },
        "action_harm_rate": {
            column.removeprefix("ndcg__"): float((frame[column] < frame["keep_ndcg"] - 1e-12).mean())
            for column in action_columns
        },
        "action_large_gain_rate": {
            column.removeprefix("ndcg__"): float((frame[column] >= frame["keep_ndcg"] + 0.05).mean())
            for column in action_columns
        },
    }
    return frame, summary


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, summary = run(args)
    frame.to_csv(output_dir / "diagnostic.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
