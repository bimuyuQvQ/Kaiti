"""把 RAGBench Parquet 转为检索实验统一格式。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIGS = (
    "covidqa",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def document_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:24]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def relevant_document_indices(row: dict[str, Any]) -> set[int]:
    """由句子键恢复相关文档位置；兼容 `[key, sentence]` 等嵌套形式。"""
    relevant_keys = {str(key) for key in (row.get("all_relevant_sentence_keys") or [])}
    if not relevant_keys:
        return set()
    indices: set[int] = set()
    for doc_index, sentence_entries in enumerate(row.get("documents_sentences") or []):
        flattened_keys: set[str] = set()
        for entry in sentence_entries or []:
            if isinstance(entry, (list, tuple)):
                flattened_keys.update(str(value) for value in entry)
            elif isinstance(entry, dict):
                flattened_keys.update(str(value) for value in entry.values())
            else:
                flattened_keys.add(str(entry))
        if flattened_keys & relevant_keys:
            indices.add(doc_index)
    return indices


def load_rows(files: list[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    required = [
        "id",
        "question",
        "documents",
        "documents_sentences",
        "all_relevant_sentence_keys",
        "dataset_name",
    ]
    for file in files:
        schema_names = set(pq.read_schema(file).names)
        missing = set(required) - schema_names
        if missing:
            raise ValueError(f"{file} 缺少字段: {sorted(missing)}")
        rows.extend(pq.read_table(file, columns=required).to_pylist())
    return rows


def convert_config(
    config: str,
    input_root: Path,
    output_root: Path,
    max_train_queries: int,
    seed: int,
) -> dict[str, Any]:
    config_input = input_root / config
    config_output = output_root / config
    corpus: dict[str, str] = {}
    split_stats: dict[str, Any] = {}

    for split in ("train", "validation", "test"):
        files = sorted(config_input.glob(f"{split}-*.parquet"))
        if not files:
            split_stats[split] = {"source_rows": 0, "kept_queries": 0, "missing": True}
            continue
        rows = load_rows(files)
        if split == "train" and len(rows) > max_train_queries:
            rows = random.Random(seed).sample(rows, max_train_queries)

        query_rows: list[dict[str, Any]] = []
        qrel_rows: list[dict[str, Any]] = []
        no_relevance = 0
        for row_index, row in enumerate(rows):
            source_id = normalize_text(row.get("id") or str(row_index))
            query_id = f"{config}:{split}:{source_id}"
            documents = [normalize_text(doc) for doc in (row.get("documents") or [])]
            doc_ids: list[str] = []
            for document in documents:
                if not document:
                    doc_ids.append("")
                    continue
                doc_id = document_id(document)
                corpus.setdefault(doc_id, document)
                doc_ids.append(doc_id)

            relevant_indices = relevant_document_indices(row)
            relevant_ids = sorted(
                {
                    doc_ids[index]
                    for index in relevant_indices
                    if index < len(doc_ids) and doc_ids[index]
                }
            )
            if not relevant_ids:
                no_relevance += 1
                continue
            query_rows.append(
                {
                    "query_id": query_id,
                    "question": normalize_text(row["question"]),
                    "config": config,
                    "split": split,
                    "source_id": source_id,
                }
            )
            qrel_rows.extend(
                {"query_id": query_id, "doc_id": doc_id, "relevance": 1}
                for doc_id in relevant_ids
            )

        query_count = write_jsonl(config_output / f"queries.{split}.jsonl", query_rows)
        qrel_count = write_jsonl(config_output / f"qrels.{split}.jsonl", qrel_rows)
        split_stats[split] = {
            "source_rows": len(rows),
            "kept_queries": query_count,
            "qrels": qrel_count,
            "rows_without_relevant_document": no_relevance,
            "source_files": [str(path) for path in files],
        }

    corpus_count = write_jsonl(
        config_output / "corpus.jsonl",
        (
            {"doc_id": doc_id, "text": text, "config": config}
            for doc_id, text in sorted(corpus.items())
        ),
    )
    return {"config": config, "corpus_documents": corpus_count, "splits": split_stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=DEFAULT_CONFIGS)
    parser.add_argument("--max-train-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = tuple(args.config or DEFAULT_CONFIGS)
    missing = [name for name in configs if not (args.input_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"缺少 RAGBench 配置目录: {', '.join(missing)}")
    reports = [
        convert_config(
            config,
            args.input_root,
            args.output_root,
            args.max_train_queries,
            args.seed,
        )
        for config in configs
    ]
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "max_train_queries": args.max_train_queries,
        "configs": reports,
    }
    manifest = args.output_root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
