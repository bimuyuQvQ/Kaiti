from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_queries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对齐 MTRAG 多种官方查询表示并生成外部候选")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="候选名=queries.jsonl 路径；可重复提供",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _parse_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"候选参数必须是 名称=路径：{spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise ValueError("候选名称不能为空")
    return name, path


def build_candidates(specs: list[str], output: str | Path) -> dict:
    if len(specs) < 1:
        raise ValueError("至少需要一个候选查询文件")
    tables: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] | None = None
    reference_ids: set[str] | None = None
    for spec in specs:
        name, path = _parse_candidate_spec(spec)
        if name in tables:
            raise ValueError(f"候选名称重复：{name}")
        queries = load_queries(path)
        table = {query.query_id: query.text for query in queries}
        current_ids = set(table)
        if reference_ids is None:
            ordered_ids = [query.query_id for query in queries]
            reference_ids = current_ids
        elif current_ids != reference_ids:
            missing = sorted(reference_ids - current_ids)
            extra = sorted(current_ids - reference_ids)
            raise ValueError(
                f"{name} 的查询 ID 集合不一致；缺少 {len(missing)} 个，多出 {len(extra)} 个"
            )
        tables[name] = table

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id in ordered_ids or []:
            for name, table in tables.items():
                handle.write(
                    json.dumps(
                        {"query_id": query_id, "action": name, "text": table[query_id]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
    summary = {
        "output": str(output_path.resolve()),
        "queries": len(ordered_ids or []),
        "actions": list(tables),
        "rows": rows,
    }
    return summary


def main() -> None:
    args = _parse_args()
    summary = build_candidates(args.candidate, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
