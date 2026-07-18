"""Collect crash-safe CURA counterfactual sample bundles.

Run from `baselines/ETC` so the released modules keep their original import
semantics.  This script creates new files only; it never deletes result data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from .canonical_runner import CanonicalTrajectoryRunner
from .jsonl_io import read_jsonl
from .manifest import build_manifest
from .rollout import audit_rollouts
from .schema import ActionRollout, RetrievedDocument, stable_id, to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_config", required=True)
    parser.add_argument("--research_config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sample", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 JSON 对象: {path}")
    return value


def write_json_exclusive(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(to_dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class ConfigNamespace:
    """Attribute config that also supports legacy ETC's membership checks."""

    def __init__(self, values: Dict[str, Any]) -> None:
        self.__dict__.update(values)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__


def load_dataset(args: ConfigNamespace) -> Any:
    from data import BIOASQ, IIRC, HotpotQA, PubmedQA, StrategyQA, WikiMultiHopQA

    classes = {
        "strategyqa": StrategyQA,
        "2wikimultihopqa": WikiMultiHopQA,
        "hotpotqa": HotpotQA,
        "iirc": IIRC,
        "bioasq_7b_yesno": BIOASQ,
        "pubmedQA": PubmedQA,
    }
    if args.dataset not in classes:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    dataset = classes[args.dataset](args.data_path)
    dataset.format(fewshot=args.fewshot)
    return dataset


def materialize_layers(run_dir: Path, bundle_paths: List[Path]) -> None:
    layer_keys = {
        "states.jsonl": "states",
        "queries.jsonl": "queries",
        "actions.jsonl": "actions",
    }
    bundles = [load_json(path) for path in bundle_paths]
    for filename, key in layer_keys.items():
        destination = run_dir / filename
        if destination.exists():
            observed = sum(1 for _ in read_jsonl(destination))
            expected = sum(len(bundle[key]) for bundle in bundles)
            if observed != expected:
                raise ValueError(f"已有 {filename} 行数为 {observed}，预期为 {expected}；拒绝覆盖")
            continue
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            for bundle in bundles:
                for row in bundle[key]:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    retrieval_rows = []
    for bundle in bundles:
        for action in bundle["actions"]:
            if action["action_type"] != "retrieve":
                continue
            for document in action.get("retrieved_documents", []):
                retrieval_rows.append(
                    {
                        "retrieval_id": stable_id(
                            "ret",
                            {
                                "action_id": action["action_id"],
                                "document_id": document["document_id"],
                                "rank": document["rank"],
                            },
                        ),
                        "qid": action["qid"],
                        "state_id": action["state_id"],
                        "action_id": action["action_id"],
                        "query_candidate_id": action["query_candidate_id"],
                        "document": document,
                    }
                )
    retrieval_path = run_dir / "retrievals.jsonl"
    if retrieval_path.exists():
        observed = sum(1 for _ in read_jsonl(retrieval_path))
        if observed != len(retrieval_rows):
            raise ValueError(
                f"已有 retrievals.jsonl 行数为 {observed}，预期为 {len(retrieval_rows)}；拒绝覆盖"
            )
    else:
        with retrieval_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in retrieval_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def action_from_dict(row: Dict[str, Any]) -> ActionRollout:
    documents = [RetrievedDocument(**document) for document in row.get("retrieved_documents", [])]
    return ActionRollout(**{**row, "retrieved_documents": documents})


def build_audit(bundle_paths: Iterable[Path]) -> Dict[str, Any]:
    actions: List[ActionRollout] = []
    expected: Dict[str, List[str]] = {}
    samples_without_states = 0
    for path in bundle_paths:
        bundle = load_json(path)
        if not bundle["states"]:
            samples_without_states += 1
        query_by_state: Dict[str, List[str]] = {}
        for query in bundle["queries"]:
            query_by_state.setdefault(query["state_id"], []).append(query["candidate_id"])
        for state in bundle["states"]:
            expected[state["state_id"]] = query_by_state.get(state["state_id"], [])
        actions.extend(action_from_dict(row) for row in bundle["actions"])
    report = audit_rollouts(actions, expected)
    value = to_dict(report)
    value["samples_without_states"] = samples_without_states
    return value


def main() -> None:
    cli = parse_args()
    legacy_config = load_json(cli.legacy_config)
    research_config = load_json(cli.research_config)
    requested_sample = cli.sample if cli.sample is not None else int(research_config.get("sample", -1))
    if requested_sample == 0 or requested_sample < -1:
        raise ValueError("sample 必须为 -1 或正数")
    effective_config = {
        "legacy": legacy_config,
        "research": {**research_config, "sample": requested_sample},
    }

    run_dir = Path(cli.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = run_dir / "sample_bundles"
    bundle_dir.mkdir(exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    manifest = build_manifest(
        effective_config,
        [cli.legacy_config, cli.research_config],
        repo_root,
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("run_id") != manifest.run_id:
            raise ValueError("运行目录已有不同配置的 manifest，拒绝混写")
    else:
        write_json_exclusive(manifest_path, manifest)

    args = ConfigNamespace(legacy_config)
    dataset = load_dataset(args)
    rows = dataset.dataset
    sample_count = len(rows) if requested_sample == -1 else min(len(rows), requested_sample)
    rows = rows.select(range(sample_count))

    pending = []
    for sample_index in range(sample_count):
        bundle_path = bundle_dir / f"sample_{sample_index:06d}.json"
        if bundle_path.exists():
            existing = load_json(bundle_path)
            if existing.get("sample_index") != sample_index or existing.get("qid") != rows[sample_index]["qid"]:
                raise ValueError(f"已有 bundle 与数据不一致: {bundle_path}")
        else:
            pending.append(sample_index)

    if pending:
        from generate import ETC

        model = ETC(args)
        runner = CanonicalTrajectoryRunner(model, dataset, research_config)
        for sample_index in tqdm(pending, desc="CURA paired rollout"):
            bundle = runner.run_sample(rows[sample_index], sample_index)
            write_json_exclusive(bundle_dir / f"sample_{sample_index:06d}.json", bundle)

    bundle_paths = [bundle_dir / f"sample_{index:06d}.json" for index in range(sample_count)]
    missing = [str(path) for path in bundle_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"仍缺少 {len(missing)} 个 sample bundle")
    materialize_layers(run_dir, bundle_paths)
    audit = build_audit(bundle_paths)
    if not audit["complete"]:
        raise RuntimeError(f"配对轨迹完整性检查失败: {audit['errors'][:5]}")
    audit_path = run_dir / "audit.json"
    if audit_path.exists():
        existing = load_json(audit_path)
        if existing != audit:
            raise ValueError("已有 audit.json 与当前审计不一致，拒绝覆盖")
    else:
        write_json_exclusive(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
