"""Build leakage-audited KEEP/RESTART supervision from matched rollout bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .compare_interventions import _document_signature
from .summarize_rollouts import _scores, load_bundle_sets


SUPERVISION_VERSION = "keep_restart_supervision_v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def _jaccard(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _document_text(documents: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{document.get('title', '')}\n{document.get('text', '')}".strip()
        for document in documents
    )


def _visible_numeric(
    question: str,
    prefix: str,
    query_text: str,
    query_source: str,
    documents: Sequence[Mapping[str, Any]],
    state_features: Mapping[str, Any],
) -> Dict[str, float]:
    evidence = _document_text(documents)
    scores = [_numeric(document.get("score")) for document in documents]
    titles = [str(document.get("title") or "") for document in documents]
    numeric = {
        "question_words": float(len(_tokens(question))),
        "prefix_words": float(len(_tokens(prefix))),
        "query_words": float(len(_tokens(query_text))),
        "evidence_words": float(len(_tokens(evidence))),
        "document_count": float(len(documents)),
        "unique_title_count": float(len(set(titles))),
        "retrieval_score_top1": scores[0] if scores else 0.0,
        "retrieval_score_mean": mean(scores) if scores else 0.0,
        "retrieval_score_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "retrieval_score_margin12": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "question_query_jaccard": _jaccard(question, query_text),
        "question_evidence_jaccard": _jaccard(question, evidence),
        "query_evidence_jaccard": _jaccard(query_text, evidence),
        "prefix_query_jaccard": _jaccard(prefix, query_text),
        "prefix_evidence_jaccard": _jaccard(prefix, evidence),
        "query_source_question": float(query_source == "question"),
        "query_source_etc_qfs": float(query_source == "etc_qfs"),
        "query_source_prefix_gap_v1": float(query_source == "prefix_gap_v1"),
    }
    for key in (
        "entropy_last",
        "entropy_mean",
        "generated_character_count",
        "generated_token_index",
        "max_attention_last",
        "max_attention_mean",
        "mt_s2_last",
        "sentence_boundaries_seen",
        "trigger_raw_attention_x_mt_s2",
        "trigger_sentence_end_word_index",
        "trigger_word_index",
    ):
        numeric[f"state_{key}"] = _numeric(state_features.get(key))
    return numeric


def build_supervision_rows(
    source_bundles: Iterable[Dict[str, Any]],
    restart_bundles: Iterable[Dict[str, Any]],
    extractor_version: str | None = "first_answer_sentence_v2",
) -> List[Dict[str, Any]]:
    sources = {(int(row["sample_index"]), str(row["qid"])): row for row in source_bundles}
    restarts = {(int(row["sample_index"]), str(row["qid"])): row for row in restart_bundles}
    if sources.keys() != restarts.keys():
        raise ValueError("source 与 restart 的样本集合不一致")

    rows: List[Dict[str, Any]] = []
    for sample_key in sorted(sources):
        source, restart = sources[sample_key], restarts[sample_key]
        if restart.get("no_etc_trigger"):
            continue
        states = {str(state["state_id"]): state for state in source["states"]}
        queries = {str(query["candidate_id"]): query for query in source["queries"]}
        source_actions = {str(action["action_id"]): action for action in source["actions"]}
        keep_actions = [action for action in restart["actions"] if action["action_type"] == "skip"]
        if len(keep_actions) != 1:
            raise ValueError(f"{sample_key} 的 KEEP 数量不是 1")
        keep = keep_actions[0]
        keep_metrics = _scores(keep, extractor_version)
        for restart_action in restart["actions"]:
            if restart_action["action_type"] != "retrieve":
                continue
            metadata = restart_action.get("generation_metadata", {})
            original_action_id = str(metadata.get("original_action_id") or "")
            if original_action_id not in source_actions:
                raise ValueError(f"缺少源 APPEND：{original_action_id}")
            append = source_actions[original_action_id]
            if append["state_id"] != restart_action["state_id"]:
                raise ValueError(f"state 不一致：{original_action_id}")
            if _document_signature(append) != _document_signature(restart_action):
                raise ValueError(f"文档不一致：{original_action_id}")
            state = states[str(append["state_id"])]
            query = queries[str(append["query_candidate_id"])]
            documents = list(append.get("retrieved_documents", []))
            evidence = _document_text(documents)
            prefix = str(state.get("prefix_text") or "")
            question = str(source["question"])
            query_text = str(query["text"])
            query_source = str(query["source"])
            restart_metrics = _scores(restart_action, extractor_version)
            append_metrics = _scores(append, extractor_version)
            keep_f1 = float(keep_metrics["f1"])
            restart_f1 = float(restart_metrics["f1"])
            delta = restart_f1 - keep_f1
            rows.append(
                {
                    "supervision_version": SUPERVISION_VERSION,
                    "sample_index": sample_key[0],
                    "qid": sample_key[1],
                    "state_id": str(append["state_id"]),
                    "query_source": query_source,
                    "query_text": query_text,
                    "document_ids": [item[0] for item in _document_signature(append)],
                    "text_full": (
                        f"[QUESTION]\n{question}\n[PREFIX]\n{prefix}\n"
                        f"[QUERY]\n{query_text}\n[EVIDENCE]\n{evidence}"
                    ),
                    "text_no_prefix": (
                        f"[QUESTION]\n{question}\n[QUERY]\n{query_text}\n"
                        f"[EVIDENCE]\n{evidence}"
                    ),
                    "visible_numeric": _visible_numeric(
                        question,
                        prefix,
                        query_text,
                        query_source,
                        documents,
                        state.get("features", {}),
                    ),
                    "labels": {
                        "keep_f1": keep_f1,
                        "append_f1": float(append_metrics["f1"]),
                        "restart_f1": restart_f1,
                        "restart_gain_over_keep": delta,
                        "keep_accuracy": float(keep_metrics["accuracy"]),
                        "restart_accuracy": float(restart_metrics["accuracy"]),
                        "restart_harm": int(
                            float(keep_metrics["accuracy"]) >= 1.0
                            and float(restart_metrics["accuracy"]) < 1.0
                        ),
                        "restart_rescue": int(
                            float(keep_metrics["accuracy"]) < 1.0
                            and float(restart_metrics["accuracy"]) >= 1.0
                        ),
                        "restart_preference": 1 if delta > 1e-12 else -1 if delta < -1e-12 else 0,
                    },
                }
            )
    if not rows:
        raise ValueError("没有可用的 KEEP/RESTART 监督行")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_supervision(
    rows: Sequence[Mapping[str, Any]], output: str | Path, metadata_output: str | Path
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    visible_keys = set(rows[0])
    forbidden = sorted(
        visible_keys
        & {
            "ground_truth",
            "ground_truth_id",
            "keep_prediction",
            "restart_prediction",
            "extracted_answer",
        }
    )
    metadata = {
        "supervision_version": SUPERVISION_VERSION,
        "rows": len(rows),
        "qids": len({str(row["qid"]) for row in rows}),
        "states": len({str(row["state_id"]) for row in rows}),
        "query_sources": sorted({str(row["query_source"]) for row in rows}),
        "harm_rows": sum(int(row["labels"]["restart_harm"]) for row in rows),
        "forbidden_top_level_fields": forbidden,
        "sha256": _sha256(output_path),
    }
    _write_json_exclusive(metadata_output, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_dir", required=True, action="append")
    parser.add_argument("--restart_run_dir", required=True, action="append")
    parser.add_argument("--extractor_version", default="first_answer_sentence_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata_output", required=True)
    args = parser.parse_args()
    extractor = None if args.extractor_version.lower() == "primary" else args.extractor_version
    rows = build_supervision_rows(
        load_bundle_sets(args.source_run_dir),
        load_bundle_sets(args.restart_run_dir),
        extractor_version=extractor,
    )
    write_supervision(rows, args.output, args.metadata_output)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "qids": len({row["qid"] for row in rows}),
                "output": args.output,
                "metadata_output": args.metadata_output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
