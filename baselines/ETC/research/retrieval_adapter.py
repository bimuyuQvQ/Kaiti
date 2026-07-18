"""Metadata-preserving BM25 adapter isolated from the legacy retriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from .schema import RetrievedDocument


class SearchBackend(Protocol):
    text_key: str
    title_key: str

    def search(self, *, index: str, body: Dict[str, Any], size: int) -> Dict[str, Any]: ...


class BeirElasticsearchBackend:
    """Lazy BEIR loader so schema/tests remain usable on CPU-only machines."""

    def __init__(self, index_name: str, hostname: str = "localhost") -> None:
        try:
            from beir.retrieval.search.lexical import BM25Search
        except ImportError as exc:
            raise RuntimeError("使用真实 BM25 检索前需要安装 ETC 所用的 beir 依赖") from exc
        wrapper = BM25Search(
            index_name=index_name,
            hostname=hostname,
            initialize=False,
            number_of_shards=1,
        ).es
        self.text_key = wrapper.text_key
        self.title_key = wrapper.title_key
        self._client = wrapper.es

    def search(self, *, index: str, body: Dict[str, Any], size: int) -> Dict[str, Any]:
        return self._client.search(
            search_type="dfs_query_then_fetch",
            index=index,
            body=body,
            size=size,
        )


class MetadataBM25:
    def __init__(
        self,
        index_name: str = "wiki",
        hostname: str = "localhost",
        backend: Optional[SearchBackend] = None,
    ) -> None:
        self.index_name = index_name
        self.backend = backend or BeirElasticsearchBackend(index_name, hostname)

    def lexical_search(self, text: str, top_hits: int, skip: int = 0) -> List[RetrievedDocument]:
        if top_hits <= 0 or skip < 0:
            raise ValueError("top_hits 必须为正数且 skip 不能为负数")
        body = {
            "query": {
                "multi_match": {
                    "query": text,
                    "type": "best_fields",
                    "fields": [self.backend.text_key, self.backend.title_key],
                    "tie_breaker": 0.5,
                }
            }
        }
        response = self.backend.search(index=self.index_name, body=body, size=skip + top_hits)
        hits = response.get("hits", {}).get("hits", [])
        documents: List[RetrievedDocument] = []
        for rank, hit in enumerate(hits[skip : skip + top_hits], start=skip + 1):
            source = hit.get("_source", {})
            text_value = source.get(self.backend.text_key, source.get("txt", ""))
            title_value = source.get(self.backend.title_key, source.get("title"))
            known = {self.backend.text_key, self.backend.title_key, "txt", "title"}
            metadata = {key: value for key, value in source.items() if key not in known}
            documents.append(
                RetrievedDocument(
                    document_id=str(hit.get("_id", "")),
                    text=str(text_value),
                    score=float(hit.get("_score", 0.0)),
                    rank=rank,
                    title=None if title_value is None else str(title_value),
                    index_name=self.index_name,
                    raw_metadata=metadata,
                )
            )
        return documents

    def __call__(self, query: str, topk: int = 1) -> List[RetrievedDocument]:
        return self.lexical_search(query, top_hits=topk)

