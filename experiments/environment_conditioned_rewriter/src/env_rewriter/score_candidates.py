"""事后检索评分并计算 best-of-N 门槛，不向生成器泄露 qrels。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .evaluate_bm25 import metrics, read_jsonl, tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=10)
    return parser.parse_args()


def score_query(
    bm25: BM25Okapi,
    doc_ids: list[str],
    query: str,
    relevant: set[str],
    cutoff: int,
) -> dict[str, float]:
    scores = bm25.get_scores(tokenize(query))
    limit = min(cutoff, len(doc_ids))
    indices = np.argpartition(scores, -limit)[-limit:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return metrics([doc_ids[index] for index in indices], relevant, cutoff)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    output_rows: list[dict[str, Any]] = []
    for config, config_rows in grouped.items():
        root = args.data_root / config
        corpus = read_jsonl(root / "corpus.jsonl")
        doc_ids = [row["doc_id"] for row in corpus]
        bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
        split = config_rows[0]["split"]
        qrels: dict[str, set[str]] = {}
        for row in read_jsonl(root / f"qrels.{split}.jsonl"):
            qrels.setdefault(row["query_id"], set()).add(row["doc_id"])
        for row in config_rows:
            texts = [row["question"], row["q0"], *row["candidates"]]
            scored = [
                {"query": text, **score_query(
                    bm25, doc_ids, text, qrels[row["query_id"]], args.cutoff
                )}
                for text in texts
            ]
            q0_score = scored[1][f"ndcg@{args.cutoff}"]
            candidate_scores = scored[2:]
            best = max(candidate_scores, key=lambda item: (
                item[f"ndcg@{args.cutoff}"],
                item[f"recall@{args.cutoff}"],
                -len(item["query"]),
            )) if candidate_scores else scored[1]
            output_rows.append({
                **row,
                "scores": scored,
                "q0_ndcg": q0_score,
                "best_ndcg": best[f"ndcg@{args.cutoff}"],
                "best_gain": best[f"ndcg@{args.cutoff}"] - q0_score,
                "best_query": best["query"],
            })
    valid = [row for row in output_rows if row["candidate_count"] == 8]
    gains = np.array([row["best_gain"] for row in valid], dtype=float)
    by_config = {}
    for config in sorted({row["config"] for row in valid}):
        subset = [row for row in valid if row["config"] == config]
        by_config[config] = {
            "count": len(subset),
            "q0_ndcg": float(np.mean([row["q0_ndcg"] for row in subset])),
            "best_ndcg": float(np.mean([row["best_ndcg"] for row in subset])),
            "gain": float(np.mean([row["best_gain"] for row in subset])),
        }
    report = {
        "schema_version": 1,
        "valid_query_count": len(valid),
        "macro_query_gain": float(gains.mean()) if len(gains) else None,
        "fraction_gain_ge_0.05": float((gains >= 0.05).mean()) if len(gains) else None,
        "h0_pass": bool(
            len(gains)
            and gains.mean() >= 0.03
            and (gains >= 0.05).mean() >= 0.20
            and sum(stats["gain"] > 0 for stats in by_config.values()) >= 2
        ),
        "by_config": by_config,
        "queries": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
