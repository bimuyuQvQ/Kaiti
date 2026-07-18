"""在已有 canonical 状态上运行 gold supporting-fact 上界实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .canonical_runner import CanonicalTrajectoryRunner
from .collect_rollouts import (
    ConfigNamespace,
    build_audit,
    load_dataset,
    load_json,
    materialize_layers,
    write_json_exclusive,
)
from .evidence_attribution import load_gold_index, normalize_text
from .legacy_adapter import build_research_etc
from .manifest import build_manifest
from .schema import (
    ActionRollout,
    CheckpointState,
    QueryCandidate,
    RetrievedDocument,
    stable_id,
    to_dict,
)
from .summarize_rollouts import load_bundle_sets


ORACLE_VERSION = "gold_supporting_facts_oracle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_config", required=True)
    parser.add_argument("--research_config", required=True)
    parser.add_argument("--source_run_dir", required=True, action="append")
    parser.add_argument("--gold_data", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--model_max_memory_gib", type=int, default=None)
    return parser.parse_args()


def select_source_bundles(
    bundles: Sequence[Dict[str, Any]], start_index: int, sample: int
) -> List[Dict[str, Any]]:
    if start_index < 0:
        raise ValueError("start_index 不能为负数")
    if sample == 0 or sample < -1:
        raise ValueError("sample 必须为 -1 或正整数")
    eligible = [bundle for bundle in bundles if int(bundle["sample_index"]) >= start_index]
    selected = eligible if sample == -1 else eligible[:sample]
    if not selected:
        raise ValueError("指定范围内没有 source bundle")
    if sample != -1 and len(selected) != sample:
        raise ValueError(f"source bundle 不足：需要 {sample}，实际 {len(selected)}")
    expected = list(range(start_index, start_index + len(selected)))
    observed = [int(bundle["sample_index"]) for bundle in selected]
    if observed != expected:
        raise ValueError(f"source bundle 的全局样本索引不连续：{observed}")
    return selected


def make_gold_candidate(state: CheckpointState, gold: Dict[str, Any]) -> QueryCandidate:
    titles = [document["title"] for document in gold["gold_documents"]]
    text = " ; ".join(titles)
    candidate_id = stable_id(
        "qry",
        {
            "version": ORACLE_VERSION,
            "qid": state.qid,
            "state_id": state.state_id,
            "gold_titles": titles,
        },
    )
    return QueryCandidate(
        qid=state.qid,
        state_id=state.state_id,
        source=ORACLE_VERSION,
        text=text,
        normalized_text=normalize_text(text),
        candidate_id=candidate_id,
        metadata={
            "oracle": True,
            "deployment_available": False,
            "evidence_granularity": "supporting_sentences",
            "gold_titles": titles,
        },
    )


def make_gold_documents(gold: Dict[str, Any]) -> List[RetrievedDocument]:
    documents = []
    for rank, document in enumerate(gold["gold_documents"], start=1):
        documents.append(
            RetrievedDocument(
                document_id=stable_id(
                    "gold_doc",
                    {
                        "version": ORACLE_VERSION,
                        "title": document["title"],
                        "sentence_ids": document["sentence_ids"],
                    },
                ),
                title=document["title"],
                text=document["text"],
                score=1.0,
                rank=rank,
                index_name="hotpotqa_gold_supporting_facts",
                raw_metadata={
                    "oracle": True,
                    "sentence_ids": document["sentence_ids"],
                },
            )
        )
    if not documents:
        raise ValueError("gold 样本没有可用支持事实文档")
    return documents


def rollout_gold_state(
    runner: CanonicalTrajectoryRunner,
    state: CheckpointState,
    candidate: QueryCandidate,
    documents: Sequence[RetrievedDocument],
    question: str,
    demo_text: str,
    ground_truth: Any,
    ground_truth_id: Any,
) -> ActionRollout:
    """严格复用 ETC 当前的文档读取、桥接句生成和原前缀续写流程。"""

    prompt = demo_text + "\nContext:\n"
    for index, document in enumerate(documents, start=1):
        prompt += f"[{index}] {document.text}\n"
    prompt += "Answer in the same format as before.\n"
    prompt += "\nQuestion:" + question + "\nAnswer:"
    retrieval_input_ids = runner.tokenizer.encode(prompt) + list(state.prefix_token_ids)
    _, regenerated, _ = runner._generate_from_ids(
        retrieval_input_ids,
        runner.max_tokens,
        stop_at_newline=True,
    )
    bridge_sentence = runner.get_top_sentence(regenerated).strip()
    bridge_token_ids = runner.tokenizer.encode(
        (" " if state.prefix_token_ids else "") + bridge_sentence,
        add_special_tokens=False,
    )
    answer_token_ids = list(state.prefix_token_ids) + list(bridge_token_ids)
    continuation_input_ids = runner.tokenizer.encode(
        demo_text + "\nQuestion:" + question + "\nAnswer:"
    ) + answer_token_ids
    remaining = max(0, runner.max_tokens - len(answer_token_ids))
    _, _, continuation_token_ids = runner._generate_from_ids(
        continuation_input_ids,
        remaining,
    )
    prediction = runner.tokenizer.decode(answer_token_ids + continuation_token_ids).strip()
    return runner._make_rollout(
        state,
        "retrieve",
        prediction,
        ground_truth,
        ground_truth_id,
        candidate,
        documents,
        {
            "retrieval_query_text": candidate.text,
            "injected_sentence": bridge_sentence,
            "evidence_source": ORACLE_VERSION,
            "oracle": True,
        },
    )


def build_oracle_bundle(
    runner: CanonicalTrajectoryRunner,
    source_bundle: Dict[str, Any],
    entry: Dict[str, Any],
    gold: Dict[str, Any],
) -> Dict[str, Any]:
    if source_bundle["qid"] != entry["qid"] or source_bundle["question"] != entry["question"]:
        raise ValueError(f"source bundle 与数据集条目不一致：{source_bundle['qid']}")
    demo_text = "\n".join(item["case"] for item in entry["demo"])
    skip_by_state: Dict[str, Dict[str, Any]] = {}
    for action in source_bundle["actions"]:
        if action["action_type"] != "skip":
            continue
        state_id = action["state_id"]
        if state_id in skip_by_state:
            raise ValueError(f"状态存在重复 skip：{state_id}")
        skip_by_state[state_id] = action
    documents = make_gold_documents(gold)
    queries: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    for state_row in source_bundle["states"]:
        state = CheckpointState(**state_row)
        if state.state_id not in skip_by_state:
            raise ValueError(f"状态缺少 source skip：{state.state_id}")
        candidate = make_gold_candidate(state, gold)
        oracle_action = rollout_gold_state(
            runner,
            state,
            candidate,
            documents,
            entry["question"],
            demo_text,
            entry["answer"],
            entry.get("answer_id"),
        )
        queries.append(to_dict(candidate))
        actions.extend([skip_by_state[state.state_id], to_dict(oracle_action)])
    copied = {
        key: value
        for key, value in source_bundle.items()
        if key not in {"bundle_version", "queries", "actions"}
    }
    return {
        "bundle_version": "cura_gold_evidence_oracle_bundle_v1",
        **copied,
        "source_bundle_version": source_bundle.get("bundle_version"),
        "oracle_version": ORACLE_VERSION,
        "queries": queries,
        "actions": actions,
    }


def _manifest_inputs(cli: argparse.Namespace) -> List[str | Path]:
    inputs: List[str | Path] = [cli.legacy_config, cli.research_config, cli.gold_data]
    for run_dir in cli.source_run_dir:
        manifest = Path(run_dir) / "manifest.json"
        if not manifest.exists():
            raise ValueError(f"source run 缺少 manifest：{manifest}")
        inputs.append(manifest)
    return inputs


def main() -> None:
    cli = parse_args()
    legacy_config = load_json(cli.legacy_config)
    research_config = load_json(cli.research_config)
    if cli.model_max_memory_gib is not None:
        if cli.model_max_memory_gib <= 0:
            raise ValueError("model_max_memory_gib 必须为正数")
        research_config = {**research_config, "model_max_memory_gib": cli.model_max_memory_gib}
    source_bundles = select_source_bundles(
        load_bundle_sets(cli.source_run_dir), cli.start_index, cli.sample
    )
    gold_index = load_gold_index(cli.gold_data)
    effective_config = {
        "oracle_version": ORACLE_VERSION,
        "legacy": legacy_config,
        "research": research_config,
        "source_run_dirs": cli.source_run_dir,
        "sample_indices": [bundle["sample_index"] for bundle in source_bundles],
    }
    run_dir = Path(cli.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = run_dir / "sample_bundles"
    bundle_dir.mkdir(exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    manifest = build_manifest(effective_config, _manifest_inputs(cli), repo_root)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("run_id") != manifest.run_id:
            raise ValueError("运行目录已有不同配置的 manifest，拒绝混写")
    else:
        write_json_exclusive(manifest_path, manifest)

    pending = [
        bundle
        for bundle in source_bundles
        if not (bundle_dir / f"sample_{int(bundle['sample_index']):06d}.json").exists()
    ]
    if pending:
        args = ConfigNamespace(legacy_config)
        dataset = load_dataset(args)
        model = build_research_etc(args, research_config)
        runner = CanonicalTrajectoryRunner(model, dataset, research_config)
        rows = dataset.dataset
        for source_bundle in pending:
            sample_index = int(source_bundle["sample_index"])
            gold = gold_index.get(normalize_text(source_bundle["question"]))
            if gold is None:
                raise ValueError(f"gold 中找不到问题：{source_bundle['question']}")
            bundle = build_oracle_bundle(
                runner,
                source_bundle,
                rows[sample_index],
                gold,
            )
            write_json_exclusive(
                bundle_dir / f"sample_{sample_index:06d}.json",
                bundle,
            )

    bundle_paths = [
        bundle_dir / f"sample_{int(bundle['sample_index']):06d}.json"
        for bundle in source_bundles
    ]
    missing = [str(path) for path in bundle_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"仍缺少 {len(missing)} 个 oracle bundle")
    materialize_layers(run_dir, bundle_paths)
    audit = build_audit(bundle_paths)
    if not audit["complete"]:
        raise RuntimeError(f"gold oracle 完整性审计失败：{audit['errors'][:5]}")
    audit_path = run_dir / "audit.json"
    if audit_path.exists():
        if load_json(audit_path) != audit:
            raise ValueError("已有 audit.json 与当前审计不一致，拒绝覆盖")
    else:
        write_json_exclusive(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        main()
