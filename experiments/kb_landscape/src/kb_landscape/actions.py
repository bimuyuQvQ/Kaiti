from __future__ import annotations

from collections import Counter

import numpy as np

from .bm25 import BM25Index, SearchResult


DEFAULT_ACTIONS = ("keep", "keywords", "prf_expand", "prf_reduce")


def _unique_in_order(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(tokens))


def keywords(query: str, index: BM25Index, max_terms: int = 8) -> str:
    tokens = _unique_in_order(index.tokenize(query))
    in_vocabulary = [token for token in tokens if index.token_idf(token) is not None]
    ranked = sorted(
        enumerate(in_vocabulary),
        key=lambda item: (-(index.token_idf(item[1]) or 0.0), item[0]),
    )
    selected = {token for _, token in ranked[:max_terms]}
    output = [token for token in in_vocabulary if token in selected]
    return " ".join(output) if output else query


def prf_expand(
    query: str,
    index: BM25Index,
    raw_result: SearchResult,
    corpus_tokens: list[list[str]],
    *,
    feedback_docs: int = 5,
    expansion_terms: int = 5,
) -> str:
    original = _unique_in_order(index.tokenize(query))
    original_set = set(original)
    weights: Counter[str] = Counter()
    selected_indices = raw_result.indices[:feedback_docs]
    selected_scores = raw_result.scores[:feedback_docs]
    if len(selected_scores) == 0:
        return query
    score_scale = max(float(np.max(selected_scores)), 1e-9)
    for rank, (doc_index, score) in enumerate(zip(selected_indices, selected_scores), start=1):
        rank_weight = max(float(score) / score_scale, 0.0) / rank
        for token, count in Counter(corpus_tokens[int(doc_index)]).items():
            if token in original_set:
                continue
            token_idf = index.token_idf(token)
            if token_idf is None:
                continue
            weights[token] += rank_weight * min(count, 3) * token_idf
    additions = [token for token, _ in weights.most_common(expansion_terms)]
    if not additions:
        return query
    return " ".join(original + additions)


def prf_reduce(
    query: str,
    index: BM25Index,
    raw_result: SearchResult,
    corpus_tokens: list[list[str]],
    *,
    feedback_docs: int = 5,
) -> str:
    original = _unique_in_order(index.tokenize(query))
    if not original or len(raw_result.indices) == 0:
        return query
    support: Counter[str] = Counter()
    for doc_index in raw_result.indices[:feedback_docs]:
        doc_vocabulary = set(corpus_tokens[int(doc_index)])
        for token in original:
            if token in doc_vocabulary:
                support[token] += 1
    supported = [token for token in original if support[token] > 0 and index.token_idf(token) is not None]
    if len(supported) < min(2, len(original)):
        return query
    return " ".join(supported)


def generate_candidates(
    query: str,
    index: BM25Index,
    raw_result: SearchResult,
    corpus_tokens: list[list[str]],
    external: dict[str, str] | None = None,
) -> dict[str, str]:
    candidates = {
        "keep": query,
        "keywords": keywords(query, index),
        "prf_expand": prf_expand(query, index, raw_result, corpus_tokens),
        "prf_reduce": prf_reduce(query, index, raw_result, corpus_tokens),
    }
    if external:
        for action, text in external.items():
            cleaned = str(text).strip()
            if cleaned:
                candidates[str(action)] = cleaned
    return candidates
