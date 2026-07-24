"""统一 BM25 检索与 nDCG@10/Recall@10/MRR@10 评测。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .prepare_ragbench import DEFAULT_CONFIGS


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(ranked: list[str], relevant: set[str], cutoff: int = 10) -> dict[str, float]:
    top = ranked[:cutoff]
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in top]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant), cutoff)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    reciprocal_rank = next(
        (1.0 / (rank + 1) for rank, doc_id in enumerate(top) if doc_id in relevant),
        0.0,
    )
    return {
        f"ndcg@{cutoff}": dcg / idcg if idcg else 0.0,
        f"recall@{cutoff}": len(set(top) & relevant) / len(relevant) if relevant else 0.0,
        f"mrr@{cutoff}": reciprocal_rank,
    }


def evaluate_config(root: Path, config: str, split: str, cutoff: int) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    config_root = root / config
    corpus = read_jsonl(config_root / "corpus.jsonl")
    queries = read_jsonl(config_root / f"queries.{split}.jsonl")
    qrels_rows = read_jsonl(config_root / f"qrels.{split}.jsonl")
    qrels: dict[str, set[str]] = {}
    for row in qrels_rows:
        qrels.setdefault(row["query_id"], set()).add(row["doc_id"])

    doc_ids = [row["doc_id"] for row in corpus]
    bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
    per_query: list[dict[str, Any]] = []
    for query in queries:
        scores = bm25.get_scores(tokenize(query["question"]))
        limit = min(cutoff, len(doc_ids))
        indices = np.argpartition(scores, -limit)[-limit:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        ranked = [doc_ids[index] for index in indices]
        query_metrics = metrics(ranked, qrels[query["query_id"]], cutoff)
        per_query.append(
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "ranked_doc_ids": ranked,
                **query_metrics,
            }
        )

    aggregate = {
        key: float(np.mean([row[key] for row in per_query]))
        for key in (f"ndcg@{cutoff}", f"recall@{cutoff}", f"mrr@{cutoff}")
    }
    return {
        "config": config,
        "split": split,
        "query_count": len(per_query),
        "corpus_count": len(corpus),
        "aggregate": aggregate,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--cutoff", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = tuple(args.config or DEFAULT_CONFIGS)
    results = [
        evaluate_config(args.data_root, config, args.split, args.cutoff)
        for config in configs
    ]
    macro = {
        key: float(np.mean([result["aggregate"][key] for result in results]))
        for key in (f"ndcg@{args.cutoff}", f"recall@{args.cutoff}", f"mrr@{args.cutoff}")
    }
    report = {
        "schema_version": 1,
        "method": "raw_question_bm25",
        "split": args.split,
        "cutoff": args.cutoff,
        "macro_average": macro,
        "configs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"macro_average": macro, "configs": [
        {"config": item["config"], **item["aggregate"]} for item in results
    ]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
