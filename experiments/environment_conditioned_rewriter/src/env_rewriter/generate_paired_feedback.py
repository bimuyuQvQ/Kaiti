"""在相同查询、相同采样噪声下生成四种匹配反馈条件的候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_bm25 import read_jsonl
from .generate_candidates import (
    SYSTEM_PROMPT,
    decode_one,
    parse_candidates,
    prompt_text,
)
from .prepare_ragbench import normalize_text


PAIRED_TEMPLATE = """Create exactly 8 distinct BM25 search queries for the question.
Do not answer the question. Across the eight queries, use a balanced mixture of:
natural rewrites, terminology-dense queries, concise keyword queries, and
document-style declarative queries.

The retrieval-feedback block is an unlabeled observation. It may be empty,
relevant, irrelevant, or produced by a different retrieval environment.
Use vocabulary or entities from it only when they genuinely help the information need.

Question:
{question}

Frozen initial query:
{q0}

Retrieval-feedback block:
{feedback}

Output exactly eight numbered lines, one query per line."""


CONDITIONS = ("none", "real", "within_corpus_shuffled", "cross_corpus_shuffled")


def stable_seed(query_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def feedback_text(
    doc_ids: list[str],
    corpus_by_id: dict[str, str],
    snippet_chars: int,
) -> str:
    if not doc_ids:
        return "[NO RETRIEVAL FEEDBACK]"
    return "\n".join(
        f"{rank}. {normalize_text(corpus_by_id[doc_id])[:snippet_chars]}"
        for rank, doc_id in enumerate(doc_ids, start=1)
        if doc_id in corpus_by_id
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-candidates", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries-per-config", type=int, default=50)
    parser.add_argument("--snippet-chars", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.source_candidates)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["config"]].append(row)
    configs = sorted(grouped)
    selected = {
        config: rows[: args.max_queries_per_config]
        for config, rows in grouped.items()
    }
    corpora: dict[str, dict[str, str]] = {}
    for config in configs:
        corpora[config] = {
            row["doc_id"]: row["text"]
            for row in read_jsonl(args.data_root / config / "corpus.jsonl")
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    completed: set[tuple[str, str]] = set()
    if args.output.exists():
        completed = {
            (row["query_id"], row["condition"])
            for row in read_jsonl(args.output)
            if len(row.get("candidates", [])) == 8
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for config_index, config in enumerate(configs):
            rows = selected[config]
            cross_config = configs[(config_index + 1) % len(configs)]
            cross_rows = selected[cross_config]
            for row_index, row in enumerate(rows):
                within_row = rows[(row_index + 1) % len(rows)]
                cross_row = cross_rows[row_index % len(cross_rows)]
                condition_doc_ids = {
                    "none": [],
                    "real": row["initial_doc_ids"],
                    "within_corpus_shuffled": within_row["initial_doc_ids"],
                    "cross_corpus_shuffled": cross_row["initial_doc_ids"],
                }
                condition_corpora = {
                    "none": corpora[config],
                    "real": corpora[config],
                    "within_corpus_shuffled": corpora[config],
                    "cross_corpus_shuffled": corpora[cross_config],
                }
                sample_seed = stable_seed(row["query_id"], args.seed)
                for condition in CONDITIONS:
                    if (row["query_id"], condition) in completed:
                        continue
                    torch.manual_seed(sample_seed)
                    torch.cuda.manual_seed_all(sample_seed)
                    feedback = feedback_text(
                        condition_doc_ids[condition],
                        condition_corpora[condition],
                        args.snippet_chars,
                    )
                    raw_generation = decode_one(
                        model,
                        tokenizer,
                        prompt_text(
                            tokenizer,
                            PAIRED_TEMPLATE.format(
                                question=row["question"],
                                q0=row["q0"],
                                feedback=feedback,
                            ),
                        ),
                        do_sample=True,
                        max_new_tokens=384,
                    )
                    candidates = parse_candidates(raw_generation)
                    record = {
                        "query_id": row["query_id"],
                        "config": config,
                        "split": row["split"],
                        "question": row["question"],
                        "q0": row["q0"],
                        "condition": condition,
                        "feedback_source_config": (
                            cross_config if condition == "cross_corpus_shuffled" else config
                        ),
                        "feedback_doc_ids": condition_doc_ids[condition],
                        "candidates": candidates,
                        "candidate_count": len(candidates),
                        "raw_generation": raw_generation,
                        "model": str(args.model),
                        "sample_seed": sample_seed,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "query_id": row["query_id"],
                                "condition": condition,
                                "candidate_count": len(candidates),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
