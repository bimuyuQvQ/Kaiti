from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(retrieved_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        relevance = float(qrels.get(doc_id, 0.0))
        if relevance > 0:
            dcg += (2.0**relevance - 1.0) / math.log2(rank + 1)
    ideal = sorted((float(value) for value in qrels.values() if value > 0), reverse=True)[:k]
    idcg = sum((2.0**value - 1.0) / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved_ids[:k])) / len(relevant)


def mrr_at_k(retrieved_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if qrels.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0
