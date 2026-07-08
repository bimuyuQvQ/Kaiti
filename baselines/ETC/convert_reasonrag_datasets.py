"""Convert ReasonRAG jsonl datasets to ETC expected layout."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def convert_hotpot(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(src_dir, "dev.jsonl")
    out: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            out.append(
                {
                    "_id": obj["id"],
                    "question": obj["question"],
                    "answer": obj["golden_answers"][0],
                }
            )
    with open(os.path.join(dst_dir, "hotpotqa-dev.json"), "w", encoding="utf-8") as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)
    print(f"hotpotqa: wrote {len(out)} examples -> {dst_dir}")


def convert_2wiki(src_dir: str, dst_dir: str, aliases_src: str | None = None) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(src_dir, "dev.jsonl")
    out: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            answer = obj["golden_answers"][0]
            out.append(
                {
                    "_id": obj["id"],
                    "question": obj["question"],
                    "answer": answer,
                    "answer_id": answer,
                }
            )
    with open(os.path.join(dst_dir, "dev.json"), "w", encoding="utf-8") as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)

    alias_out = os.path.join(dst_dir, "id_aliases.json")
    if aliases_src and os.path.exists(aliases_src):
        with open(aliases_src, "r", encoding="utf-8") as fin, open(
            alias_out, "w", encoding="utf-8"
        ) as fout:
            fout.write(fin.read())
        print(f"2wiki: copied id_aliases from {aliases_src}")
    else:
        with open(alias_out, "w", encoding="utf-8") as fout:
            for item in out:
                rec = {"Q_id": item["answer_id"], "aliases": [item["answer"]], "demonyms": []}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"2wiki: wrote fallback id_aliases (exact answer only) -> {alias_out}")

    print(f"2wiki: wrote {len(out)} examples -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reasonrag_root",
        type=str,
        default="../ReasonRAG-main",
        help="ReasonRAG repo root",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="ETC data root",
    )
    parser.add_argument(
        "--aliases_src",
        type=str,
        default="",
        help="Optional official id_aliases.json for 2wiki",
    )
    args = parser.parse_args()

    rr = os.path.abspath(args.reasonrag_root)
    out_root = os.path.abspath(args.output_root)
    aliases = args.aliases_src or None

    convert_hotpot(
        os.path.join(rr, "dataset/hotpotqa"),
        os.path.join(out_root, "hotpotqa"),
    )
    convert_2wiki(
        os.path.join(rr, "dataset/2wikimultihopqa"),
        os.path.join(out_root, "2wikimultihopqa"),
        aliases_src=aliases,
    )


if __name__ == "__main__":
    main()
