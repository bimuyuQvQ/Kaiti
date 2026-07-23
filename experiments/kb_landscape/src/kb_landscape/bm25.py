from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


class BM25Index:
    """与主基线口径一致的轻量 BM25 稀疏矩阵实现。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, max_features: int | None = None):
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
            max_features=max_features,
        )
        self.doc_weights: sp.csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.doc_lengths: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "BM25Index":
        counts = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = counts.shape[0]
        self.doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = max(float(self.doc_lengths.mean()), 1e-9)
        document_frequency = np.asarray(counts.getnnz(axis=0)).ravel()
        self.idf = np.log((n_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).astype(
            np.float32
        )

        coo = counts.tocoo()
        length_norm = 1.0 - self.b + self.b * self.doc_lengths[coo.row] / average_length
        values = (coo.data * (self.k1 + 1.0)) / (coo.data + self.k1 * length_norm)
        self.doc_weights = sp.csr_matrix((values, (coo.row, coo.col)), shape=counts.shape, dtype=np.float32)
        return self

    @property
    def vocabulary(self) -> dict[str, int]:
        return self.vectorizer.vocabulary_

    def tokenize(self, text: str) -> list[str]:
        return list(self.vectorizer.build_analyzer()(text))

    def token_idf(self, token: str) -> float | None:
        if self.idf is None:
            raise RuntimeError("BM25Index 尚未 fit")
        index = self.vocabulary.get(token)
        if index is None:
            return None
        return float(self.idf[index])

    def search(self, query: str, top_k: int = 10) -> SearchResult:
        if self.doc_weights is None or self.idf is None:
            raise RuntimeError("BM25Index 尚未 fit")
        query_counts = self.vectorizer.transform([query]).tocsr()
        if query_counts.nnz == 0:
            size = min(top_k, self.doc_weights.shape[0])
            return SearchResult(
                indices=np.arange(size, dtype=np.int64),
                scores=np.zeros(size, dtype=np.float32),
            )
        query_counts.data = np.ones_like(query_counts.data, dtype=np.float32)
        query_weights = query_counts.multiply(self.idf)
        scores = np.asarray(self.doc_weights @ query_weights.T).ravel()
        size = min(top_k, scores.size)
        if size == scores.size:
            indices = np.argsort(-scores, kind="stable")
        else:
            partition = np.argpartition(scores, -size)[-size:]
            indices = partition[np.argsort(-scores[partition], kind="stable")]
        return SearchResult(indices=indices.astype(np.int64), scores=scores[indices].astype(np.float32))
