from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .bm25 import BM25Index, SearchResult


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def raw_landscape_features(
    query: str,
    index: BM25Index,
    result: SearchResult,
    corpus_tokens: list[list[str]],
) -> dict[str, float]:
    query_tokens = index.tokenize(query)
    query_unique = set(query_tokens)
    idf_values = [value for token in query_unique if (value := index.token_idf(token)) is not None]
    oov_count = sum(index.token_idf(token) is None for token in query_unique)
    scores = result.scores.astype(np.float64)
    nonnegative = np.maximum(scores, 0.0)
    if len(nonnegative) and float(nonnegative.sum()) > 0:
        probabilities = nonnegative / nonnegative.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    else:
        normalized_entropy = 0.0

    top_token_sets = [set(corpus_tokens[int(doc_index)]) for doc_index in result.indices]
    pairwise_jaccard = [_jaccard(left, right) for left, right in combinations(top_token_sets, 2)]
    coverages = [
        len(query_unique & token_set) / len(query_unique) if query_unique else 0.0
        for token_set in top_token_sets
    ]
    document_lengths = [float(len(corpus_tokens[int(doc_index)])) for doc_index in result.indices]
    if len(scores) > 1:
        slope = float(np.polyfit(np.arange(len(scores), dtype=np.float64), scores, 1)[0])
        margin12 = float(scores[0] - scores[1])
    else:
        slope = 0.0
        margin12 = float(scores[0]) if len(scores) else 0.0

    return {
        "feat_query_token_count": float(len(query_tokens)),
        "feat_query_unique_count": float(len(query_unique)),
        "feat_query_oov_ratio": oov_count / len(query_unique) if query_unique else 0.0,
        "feat_query_idf_mean": _safe_mean(idf_values),
        "feat_query_idf_std": _safe_std(idf_values),
        "feat_query_idf_max": max(idf_values, default=0.0),
        "feat_score_top1": float(scores[0]) if len(scores) else 0.0,
        "feat_score_margin12": margin12,
        "feat_score_mean": float(scores.mean()) if len(scores) else 0.0,
        "feat_score_std": float(scores.std()) if len(scores) else 0.0,
        "feat_score_range": float(scores.max() - scores.min()) if len(scores) else 0.0,
        "feat_score_slope": slope,
        "feat_score_entropy": normalized_entropy,
        "feat_score_nonzero_ratio": float(np.mean(scores > 0)) if len(scores) else 0.0,
        "feat_top_doc_length_mean": _safe_mean(document_lengths),
        "feat_top_doc_length_std": _safe_std(document_lengths),
        "feat_top_doc_pairwise_jaccard": _safe_mean(pairwise_jaccard),
        "feat_query_doc_coverage_mean": _safe_mean(coverages),
        "feat_query_doc_coverage_max": max(coverages, default=0.0),
    }


def probe_stability_features(results: dict[str, SearchResult]) -> dict[str, float]:
    action_sets = {action: set(result.indices.tolist()) for action, result in results.items()}
    overlaps = [
        _jaccard(action_sets[left], action_sets[right])
        for left, right in combinations(action_sets, 2)
    ]
    top1_values = [int(result.indices[0]) for result in results.values() if len(result.indices)]
    top1_consensus = max((top1_values.count(value) for value in set(top1_values)), default=0)
    union = set().union(*action_sets.values()) if action_sets else set()
    return {
        "feat_probe_overlap_mean": _safe_mean(overlaps),
        "feat_probe_overlap_min": min(overlaps, default=1.0),
        "feat_probe_union_size": float(len(union)),
        "feat_probe_top1_consensus": top1_consensus / len(top1_values) if top1_values else 0.0,
    }
