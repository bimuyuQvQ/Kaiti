"""评分匹配反馈条件，输出查询级结果供配对 bootstrap 分析。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .evaluate_bm25 import read_jsonl, tokenize
from .score_candidates import score_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.candidates)
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
    scored_rows: list[dict[str, Any]] = []
    for config, config_rows in by_config.items():
        root = args.data_root / config
        corpus = read_jsonl(root / "corpus.jsonl")
        doc_ids = [row["doc_id"] for row in corpus]
        bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
        split = config_rows[0]["split"]
        qrels: dict[str, set[str]] = defaultdict(set)
        for qrel in read_jsonl(root / f"qrels.{split}.jsonl"):
            qrels[qrel["query_id"]].add(qrel["doc_id"])
        for row in config_rows:
            q0_score = score_query(
                bm25, doc_ids, row["q0"], qrels[row["query_id"]], args.cutoff
            )[f"ndcg@{args.cutoff}"]
            candidate_scores = [
                score_query(
                    bm25, doc_ids, query, qrels[row["query_id"]], args.cutoff
                )[f"ndcg@{args.cutoff}"]
                for query in row["candidates"]
            ]
            oracle = max([q0_score, *candidate_scores])
            scored_rows.append(
                {
                    "query_id": row["query_id"],
                    "config": config,
                    "condition": row["condition"],
                    "candidate_count": row["candidate_count"],
                    "q0_ndcg": q0_score,
                    "candidate_mean_ndcg": float(np.mean(candidate_scores)),
                    "candidate_ndcg": candidate_scores,
                    "oracle_ndcg": oracle,
                    "oracle_gain": oracle - q0_score,
                }
            )

    aggregates: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in scored_rows}):
        subset = [
            row for row in scored_rows
            if row["condition"] == condition and row["candidate_count"] == 8
        ]
        aggregates[condition] = {
            "valid_queries": len(subset),
            "candidate_mean_ndcg": float(
                np.mean([row["candidate_mean_ndcg"] for row in subset])
            ),
            "oracle_ndcg": float(np.mean([row["oracle_ndcg"] for row in subset])),
            "oracle_gain": float(np.mean([row["oracle_gain"] for row in subset])),
            "fraction_oracle_gain_ge_0.05": float(
                np.mean([row["oracle_gain"] >= 0.05 for row in subset])
            ),
        }
    report = {
        "schema_version": 1,
        "aggregates": aggregates,
        "queries": scored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
