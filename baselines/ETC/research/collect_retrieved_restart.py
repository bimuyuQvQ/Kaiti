"""在首次 ETC 触发状态上用已有真实检索文档做无前缀重启诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .canonical_runner import CanonicalTrajectoryRunner
from .collect_gold_oracle import generate_restart_prediction, select_source_bundles
from .collect_rollouts import (
    ConfigNamespace,
    build_audit,
    load_dataset,
    load_json,
    materialize_layers,
    write_json_exclusive,
)
from .legacy_adapter import build_research_etc
from .manifest import build_manifest
from .schema import CheckpointState, QueryCandidate, RetrievedDocument, stable_id, to_dict
from .summarize_rollouts import load_bundle_sets


RESTART_VERSION = "retrieved_evidence_restart_at_etc_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_config", required=True)
    parser.add_argument("--research_config", required=True)
    parser.add_argument("--source_run_dir", required=True, action="append")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--model_max_memory_gib", type=int, default=None)
    return parser.parse_args()


def select_first_etc_state(states: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    matches = [state for state in states if state["checkpoint_type"] == "first_etc_trigger"]
    if len(matches) > 1:
        raise ValueError("同一样本出现多个 first_etc_trigger 状态")
    return matches[0] if matches else None


def make_restart_candidate(
    state: CheckpointState, original: Dict[str, Any]
) -> QueryCandidate:
    source = f"retrieved_restart_{original['source']}_v1"
    candidate_id = stable_id(
        "qry",
        {
            "version": RESTART_VERSION,
            "state_id": state.state_id,
            "original_candidate_id": original["candidate_id"],
        },
    )
    return QueryCandidate(
        qid=state.qid,
        state_id=state.state_id,
        source=source,
        text=original["text"],
        normalized_text=original["normalized_text"],
        candidate_id=candidate_id,
        metadata={
            "intervention": "restart_from_retrieved_evidence",
            "original_candidate_id": original["candidate_id"],
            "original_query_source": original["source"],
            "deployment_available": True,
        },
    )


def build_restart_bundle(
    runner: CanonicalTrajectoryRunner,
    source_bundle: Dict[str, Any],
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    if source_bundle["qid"] != entry["qid"] or source_bundle["question"] != entry["question"]:
        raise ValueError(f"source bundle 与数据集条目不一致：{source_bundle['qid']}")
    selected_state_row = select_first_etc_state(source_bundle["states"])
    copied = {
        key: value
        for key, value in source_bundle.items()
        if key not in {"bundle_version", "states", "queries", "actions"}
    }
    if selected_state_row is None:
        return {
            "bundle_version": "cura_retrieved_restart_bundle_v1",
            **copied,
            "source_bundle_version": source_bundle.get("bundle_version"),
            "restart_version": RESTART_VERSION,
            "no_etc_trigger": True,
            "states": [],
            "queries": [],
            "actions": [],
        }

    state = CheckpointState(**selected_state_row)
    original_queries = {query["candidate_id"]: query for query in source_bundle["queries"]}
    state_actions = [
        action for action in source_bundle["actions"] if action["state_id"] == state.state_id
    ]
    skips = [action for action in state_actions if action["action_type"] == "skip"]
    if len(skips) != 1:
        raise ValueError(f"ETC 状态 skip 数量不是 1：{state.state_id}")
    retrieves = [action for action in state_actions if action["action_type"] == "retrieve"]
    if not retrieves:
        raise ValueError(f"ETC 状态没有真实检索动作：{state.state_id}")

    demo_text = "\n".join(item["case"] for item in entry["demo"])
    queries: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = [skips[0]]
    for original_action in retrieves:
        original_query = original_queries[original_action["query_candidate_id"]]
        candidate = make_restart_candidate(state, original_query)
        documents = [
            RetrievedDocument(**document)
            for document in original_action.get("retrieved_documents", [])
        ]
        if not documents:
            raise ValueError(f"真实检索动作没有文档：{original_action.get('action_id')}")
        prediction = generate_restart_prediction(
            runner,
            documents,
            entry["question"],
            demo_text,
        )
        action = runner._make_rollout(
            state,
            "retrieve",
            prediction,
            entry["answer"],
            entry.get("answer_id"),
            candidate,
            documents,
            {
                "retrieval_query_text": candidate.text,
                "injected_sentence": runner.get_top_sentence(prediction).strip(),
                "intervention": "restart_from_retrieved_evidence",
                "original_action_id": original_action["action_id"],
                "original_query_candidate_id": original_query["candidate_id"],
                "restart_version": RESTART_VERSION,
            },
        )
        queries.append(to_dict(candidate))
        actions.append(to_dict(action))
    return {
        "bundle_version": "cura_retrieved_restart_bundle_v1",
        **copied,
        "source_bundle_version": source_bundle.get("bundle_version"),
        "restart_version": RESTART_VERSION,
        "no_etc_trigger": False,
        "states": [selected_state_row],
        "queries": queries,
        "actions": actions,
    }


def _manifest_inputs(cli: argparse.Namespace) -> List[str | Path]:
    inputs: List[str | Path] = [cli.legacy_config, cli.research_config]
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
    effective_config = {
        "restart_version": RESTART_VERSION,
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
        if load_json(manifest_path).get("run_id") != manifest.run_id:
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
            bundle = build_restart_bundle(runner, source_bundle, rows[sample_index])
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
        raise RuntimeError(f"仍缺少 {len(missing)} 个 restart bundle")
    materialize_layers(run_dir, bundle_paths)
    audit = build_audit(bundle_paths)
    if not audit["complete"]:
        raise RuntimeError(f"restart 完整性审计失败：{audit['errors'][:5]}")
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
