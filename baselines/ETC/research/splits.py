"""Dataset-order-independent train/calibration/test assignment."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence


DEFAULT_SPLIT_RATIOS = {"train": 0.7, "calibration": 0.15, "test": 0.15}


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if list(ratios) != ["train", "calibration", "test"]:
        raise ValueError("划分必须按 train、calibration、test 顺序给出")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("每个划分比例必须为正数")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("划分比例之和必须为 1")


def split_for_qid(
    qid: str,
    salt: str = "etc_cura_split_v1",
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> str:
    _validate_ratios(ratios)
    digest = hashlib.sha256(f"{salt}\0{qid}".encode("utf-8")).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        if point < cumulative:
            return name
    return "test"


def assign_splits(qids: Iterable[str], salt: str = "etc_cura_split_v1") -> Dict[str, str]:
    qid_list = list(qids)
    duplicates = [qid for qid, count in Counter(qid_list).items() if count > 1]
    if duplicates:
        raise ValueError(f"qid 重复，无法安全划分: {duplicates[:5]}")
    return {qid: split_for_qid(qid, salt=salt) for qid in qid_list}


def validate_disjoint(split_to_qids: Mapping[str, Sequence[str]]) -> None:
    owner: Dict[str, str] = {}
    for split, qids in split_to_qids.items():
        for qid in qids:
            if qid in owner:
                raise ValueError(f"qid={qid} 同时出现在 {owner[qid]} 和 {split}")
            owner[qid] = split

