"""用 HotpotQA 支持事实归因已有反事实检索动作，不运行模型。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .summarize_rollouts import _benefit_bucket, _scores, load_bundle_sets


AUDIT_VERSION = "query_evidence_attribution_v1"


def normalize_text(value: Any) -> str:
    """适合标题和证据子串匹配的保守 Unicode 规范化。"""

    text = unicodedata.normalize("NFKC", str(value or "")).replace("_", " ").casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _unpack_gold_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = []
        for item in payload["rows"]:
            row = item.get("row") if isinstance(item, dict) else None
            if not isinstance(row, dict):
                raise ValueError("HF rows 包含缺失或非法的 row")
            rows.append(row)
        return rows
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("不支持的 HotpotQA gold 格式")


def _context_map(row: Mapping[str, Any]) -> Dict[str, List[str]]:
    context = row.get("context")
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        if len(titles) != len(sentence_groups):
            raise ValueError("context.title 与 context.sentences 长度不一致")
        pairs = zip(titles, sentence_groups)
    elif isinstance(context, list):
        pairs = []
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("官方 context 条目格式非法")
            pairs.append((item[0], item[1]))
    else:
        raise ValueError("gold 样本缺少 context")
    result: Dict[str, List[str]] = {}
    for title, sentences in pairs:
        key = normalize_text(title)
        if key in result:
            raise ValueError(f"context 出现规范化后重复标题: {title}")
        result[key] = [str(sentence) for sentence in sentences]
    return result


def _support_pairs(row: Mapping[str, Any]) -> List[Tuple[str, int]]:
    facts = row.get("supporting_facts")
    if isinstance(facts, dict):
        titles = facts.get("title", [])
        sentence_ids = facts.get("sent_id", [])
        if len(titles) != len(sentence_ids):
            raise ValueError("supporting_facts.title 与 sent_id 长度不一致")
        raw_pairs = zip(titles, sentence_ids)
    elif isinstance(facts, list):
        raw_pairs = facts
    else:
        raise ValueError("gold 样本缺少 supporting_facts")
    result = []
    for item in raw_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("supporting_facts 条目格式非法")
        result.append((str(item[0]), int(item[1])))
    return result


def build_gold_index(payload: Any) -> Dict[str, Dict[str, Any]]:
    """按规范化问题建立索引，并解析金标题与金支持句。"""

    index: Dict[str, Dict[str, Any]] = {}
    for row in _unpack_gold_payload(payload):
        question = str(row.get("question", ""))
        question_key = normalize_text(question)
        if not question_key:
            raise ValueError("gold 样本的问题为空")
        if question_key in index:
            raise ValueError(f"gold 中出现重复问题: {question}")
        contexts = _context_map(row)
        pairs = _support_pairs(row)
        titles: List[str] = []
        normalized_titles: List[str] = []
        sentences: List[str] = []
        normalized_sentences: List[str] = []
        for title, sentence_id in pairs:
            title_key = normalize_text(title)
            if title_key not in contexts:
                raise ValueError(f"支持事实标题不在 context 中: {title}")
            if sentence_id < 0 or sentence_id >= len(contexts[title_key]):
                raise ValueError(f"支持句下标越界: {title}[{sentence_id}]")
            sentence = contexts[title_key][sentence_id]
            if title_key not in normalized_titles:
                titles.append(title)
                normalized_titles.append(title_key)
            sentence_key = normalize_text(sentence)
            if sentence_key and sentence_key not in normalized_sentences:
                sentences.append(sentence)
                normalized_sentences.append(sentence_key)
        index[question_key] = {
            "question": question,
            "answer": str(row.get("answer", "")),
            "gold_titles": titles,
            "normalized_gold_titles": normalized_titles,
            "gold_support_sentences": sentences,
            "normalized_gold_support_sentences": normalized_sentences,
        }
    return index


def load_gold_index(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return build_gold_index(json.load(handle))


def _first_rank(documents: Sequence[Mapping[str, Any]], gold_titles: set[str]) -> int | None:
    ranks = [
        int(document.get("rank", position + 1))
        for position, document in enumerate(documents)
        if normalize_text(document.get("title")) in gold_titles
    ]
    return min(ranks) if ranks else None


def _contains_any(document_text: str, needles: Sequence[str]) -> bool:
    return any(needle and needle in document_text for needle in needles)


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "count": 0,
            "sample_count": 0,
            "mean_benefit": 0.0,
            "any_gold_title_hit_rate": 0.0,
            "full_gold_title_hit_rate": 0.0,
            "support_sentence_hit_rate": 0.0,
            "answer_hit_rate": 0.0,
            "mean_gold_title_recall": 0.0,
        }
    mean = lambda key: sum(float(row[key]) for row in rows) / count
    return {
        "count": count,
        "sample_count": len({(row["sample_index"], row["qid"]) for row in rows}),
        "mean_benefit": mean("benefit"),
        "any_gold_title_hit_rate": mean("any_gold_title_hit"),
        "full_gold_title_hit_rate": mean("full_gold_title_hit"),
        "support_sentence_hit_rate": mean("support_sentence_hit"),
        "answer_hit_rate": mean("answer_hit"),
        "mean_gold_title_recall": mean("gold_title_recall"),
    }


def _failure_label(row: Mapping[str, Any]) -> str:
    if not row["any_gold_title_hit"]:
        return "gold_title_miss"
    if not row["support_sentence_hit"]:
        return "gold_title_hit_but_support_sentence_miss"
    if row["benefit_bucket"] == "positive":
        return "support_sentence_hit_and_gain"
    if row["benefit_bucket"] == "negative":
        return "support_sentence_hit_but_harm"
    return "support_sentence_hit_but_no_gain"


def attribute_bundles(
    bundles: Iterable[Dict[str, Any]],
    gold_index: Mapping[str, Dict[str, Any]],
    metric: str = "f1",
    extractor_version: str | None = "first_answer_sentence_v2",
) -> Dict[str, Any]:
    bundle_list = list(bundles)
    rows: List[Dict[str, Any]] = []
    matched_questions: set[str] = set()
    for bundle in bundle_list:
        question_key = normalize_text(bundle.get("question"))
        if question_key not in gold_index:
            raise ValueError(f"gold 中找不到结果包问题: {bundle.get('question')}")
        if question_key in matched_questions:
            raise ValueError(f"结果包中出现重复问题: {bundle.get('question')}")
        matched_questions.add(question_key)
        gold = gold_index[question_key]
        gold_titles = set(gold["normalized_gold_titles"])
        states = {state["state_id"]: state for state in bundle["states"]}
        queries = {query["candidate_id"]: query for query in bundle["queries"]}
        skip_by_state = {
            action["state_id"]: action
            for action in bundle["actions"]
            if action["action_type"] == "skip"
        }
        if len(skip_by_state) != len(states):
            raise ValueError(f"样本 {bundle['qid']} 的状态未各自对应唯一 skip")
        for action in bundle["actions"]:
            if action["action_type"] != "retrieve":
                continue
            state_id = action["state_id"]
            query = queries[action["query_candidate_id"]]
            retrieve_score = float(_scores(action, extractor_version)[metric])
            skip_score = float(_scores(skip_by_state[state_id], extractor_version)[metric])
            benefit = retrieve_score - skip_score
            documents = action.get("retrieved_documents", [])
            retrieved_titles = [str(document.get("title") or "") for document in documents]
            retrieved_title_keys = {normalize_text(title) for title in retrieved_titles if title}
            hit_titles = gold_titles & retrieved_title_keys
            merged_document_text = normalize_text(
                " ".join(str(document.get("text") or "") for document in documents)
            )
            support_hit = _contains_any(
                merged_document_text, gold["normalized_gold_support_sentences"]
            )
            answer_key = normalize_text(bundle.get("ground_truth") or gold.get("answer"))
            row = {
                "sample_index": bundle["sample_index"],
                "qid": bundle["qid"],
                "question": bundle["question"],
                "state_id": state_id,
                "checkpoint_type": states[state_id]["checkpoint_type"],
                "query_source": query["source"],
                "query_text": query["text"],
                "benefit": benefit,
                "benefit_bucket": _benefit_bucket(benefit),
                "skip_score": skip_score,
                "retrieve_score": retrieve_score,
                "gold_titles": gold["gold_titles"],
                "retrieved_titles": retrieved_titles,
                "hit_gold_titles": sorted(hit_titles),
                "any_gold_title_hit": bool(hit_titles),
                "full_gold_title_hit": bool(gold_titles) and gold_titles <= retrieved_title_keys,
                "gold_title_recall": len(hit_titles) / len(gold_titles) if gold_titles else 0.0,
                "first_gold_title_rank": _first_rank(documents, gold_titles),
                "support_sentence_hit": support_hit,
                "answer_hit": bool(answer_key and answer_key in merged_document_text),
                "injected_sentence": action.get("generation_metadata", {}).get("injected_sentence"),
            }
            row["attribution_label"] = _failure_label(row)
            rows.append(row)

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["benefit_bucket"]].append(row)
        by_source[row["query_source"]].append(row)
        by_label[row["attribution_label"]].append(row)
    return {
        "audit_version": AUDIT_VERSION,
        "metric": metric,
        "extractor_version": extractor_version or "primary",
        "samples": len(bundle_list),
        "retrieve_actions": len(rows),
        "matched_gold_questions": len(matched_questions),
        "coverage_complete": len(matched_questions) == len(bundle_list),
        "overall": _summarize_rows(rows),
        "by_benefit_bucket": {
            key: _summarize_rows(by_bucket.get(key, []))
            for key in ("positive", "zero", "negative")
        },
        "by_query_source": {
            key: _summarize_rows(group) for key, group in sorted(by_source.items())
        },
        "attribution_counts": {
            key: {"actions": len(group), "samples": len({row["sample_index"] for row in group})}
            for key, group in sorted(by_label.items())
        },
        "diagnostic_cases": rows,
        "interpretation_caveat": (
            "标题命中不保证检索到正确 chunk；支持句字符串命中是更强但仍不完美的证据代理。"
            "动作共享样本和状态，动作级比例不能当作独立样本显著性检验。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, action="append")
    parser.add_argument("--gold_data", required=True)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--extractor_version", default="first_answer_sentence_v2")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = attribute_bundles(
        load_bundle_sets(args.run_dir),
        load_gold_index(args.gold_data),
        metric=args.metric,
        extractor_version=args.extractor_version or None,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
