"""用干净的原始指令模型生成统一 q0 与八类自由查询候选。"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_bm25 import read_jsonl, tokenize
from .prepare_ragbench import DEFAULT_CONFIGS, normalize_text


SYSTEM_PROMPT = (
    "You formulate search-engine queries. Return only the requested query text or "
    "numbered query list. Never answer the question."
)

Q0_TEMPLATE = """Rewrite the question as one concise standalone search query.
Do not answer it. Output only the query.

Question: {question}"""

CANDIDATE_TEMPLATE = """Create exactly 8 distinct search queries for the question.
The queries will be sent to BM25 over the target corpus. Do not answer the question.

Required diversity:
1-2: natural semantic rewrites
3-4: keyword- and terminology-dense queries
5-6: document-style declarative queries
7-8: feedback-aware queries using useful vocabulary or entities from the initial results,
     while ignoring results that appear irrelevant

Question:
{question}

Frozen initial query:
{q0}

Initial BM25 results from the target corpus (unlabeled; they may be wrong):
{feedback}

Output exactly eight numbered lines, one query per line."""


def prompt_text(tokenizer: Any, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def decode_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    do_sample: bool,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**encoded, **kwargs)
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)、:])\s*", "", line).strip()
        cleaned = cleaned.strip("\"'`")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates[:8]


def retrieve_feedback(
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    query: str,
    top_k: int,
    snippet_chars: int,
) -> tuple[list[str], str]:
    scores = bm25.get_scores(tokenize(query))
    limit = min(top_k, len(corpus))
    indices = np.argpartition(scores, -limit)[-limit:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    doc_ids = [corpus[index]["doc_id"] for index in indices]
    feedback = "\n".join(
        f"{rank}. {normalize_text(corpus[index]['text'])[:snippet_chars]}"
        for rank, index in enumerate(indices, start=1)
    )
    return doc_ids, feedback


def selected_queries(
    rows: list[dict[str, Any]],
    max_queries: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_queries <= 0 or len(rows) <= max_queries:
        return rows
    return random.Random(seed).sample(rows, max_queries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-queries-per-config", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    completed: set[str] = set()
    if args.output.exists():
        completed = {
            row["query_id"] for row in read_jsonl(args.output)
            if len(row.get("candidates", [])) == 8
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configs = tuple(args.config or DEFAULT_CONFIGS)
    with args.output.open("a", encoding="utf-8") as handle:
        for config_index, config in enumerate(configs):
            root = args.data_root / config
            corpus = read_jsonl(root / "corpus.jsonl")
            bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
            queries = selected_queries(
                read_jsonl(root / f"queries.{args.split}.jsonl"),
                args.max_queries_per_config,
                args.seed + config_index,
            )
            for query in queries:
                if query["query_id"] in completed:
                    continue
                started = time.perf_counter()
                q0_raw = decode_one(
                    model,
                    tokenizer,
                    prompt_text(tokenizer, Q0_TEMPLATE.format(question=query["question"])),
                    do_sample=False,
                    max_new_tokens=64,
                )
                q0 = normalize_text(q0_raw.splitlines()[0])
                initial_doc_ids, feedback = retrieve_feedback(
                    bm25, corpus, q0, args.top_k, args.snippet_chars
                )
                raw_generation = decode_one(
                    model,
                    tokenizer,
                    prompt_text(
                        tokenizer,
                        CANDIDATE_TEMPLATE.format(
                            question=query["question"],
                            q0=q0,
                            feedback=feedback,
                        ),
                    ),
                    do_sample=True,
                    max_new_tokens=384,
                )
                candidates = parse_candidates(raw_generation)
                record = {
                    "query_id": query["query_id"],
                    "config": config,
                    "split": args.split,
                    "question": query["question"],
                    "q0": q0,
                    "initial_doc_ids": initial_doc_ids,
                    "candidates": candidates,
                    "candidate_count": len(candidates),
                    "raw_generation": raw_generation,
                    "model": str(args.model),
                    "seed": args.seed,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "query_id": query["query_id"],
                            "candidate_count": len(candidates),
                            "elapsed_seconds": record["elapsed_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
