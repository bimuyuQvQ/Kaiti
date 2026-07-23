from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        if self.title:
            return f"{self.title}\n{self.text}"
        return self.text


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class BeirDataset:
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, float]]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            yield value


def _pick(record: dict, names: tuple[str, ...], *, required: bool = True) -> str:
    for name in names:
        if name in record and record[name] is not None:
            return str(record[name])
    if required:
        raise KeyError(f"缺少字段，候选字段为：{', '.join(names)}")
    return ""


def load_corpus(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    for record in _read_jsonl(path):
        doc_id = _pick(record, ("_id", "doc_id", "id"))
        if doc_id in seen:
            raise ValueError(f"语料中存在重复文档 ID：{doc_id}")
        seen.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                title=_pick(record, ("title",), required=False),
                text=_pick(record, ("text", "contents", "body")),
            )
        )
    if not documents:
        raise ValueError(f"语料为空：{path}")
    return documents


def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for record in _read_jsonl(path):
        query_id = _pick(record, ("_id", "query_id", "id"))
        if query_id in seen:
            raise ValueError(f"查询中存在重复 ID：{query_id}")
        seen.add(query_id)
        queries.append(Query(query_id=query_id, text=_pick(record, ("text", "query", "question"))))
    if not queries:
        raise ValueError(f"查询为空：{path}")
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else None
        if delimiter:
            rows = csv.reader(handle, delimiter=delimiter)
        else:
            rows = (line.split() for line in handle)

        first_data_row = True
        for row in rows:
            if not row or all(not str(cell).strip() for cell in row):
                continue
            cells = [str(cell).strip() for cell in row]
            if len(cells) < 3:
                raise ValueError(f"qrels 行至少需要三列：{cells}")
            if first_data_row:
                first_data_row = False
                header = "query" in cells[0].lower() or "score" in cells[-1].lower()
                if header:
                    continue
            query_id, doc_id, score_text = cells[0], cells[1], cells[2]
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(f"qrels 相关性分数无法解析：{cells}") from exc
            if score > 0:
                qrels.setdefault(query_id, {})[doc_id] = score
    if not qrels:
        raise ValueError(f"qrels 为空：{path}")
    return qrels


def load_beir_dataset(root: str | Path, split: str = "test") -> BeirDataset:
    root_path = Path(root)
    corpus_path = root_path / "corpus.jsonl"
    query_path = root_path / "queries.jsonl"
    qrels_path = root_path / "qrels" / f"{split}.tsv"
    for path in (corpus_path, query_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少 BEIR 文件：{path}")

    documents = load_corpus(corpus_path)
    queries = load_queries(query_path)
    qrels = load_qrels(qrels_path)
    known_docs = {document.doc_id for document in documents}
    filtered_queries = [query for query in queries if query.query_id in qrels]
    missing_docs = sorted(
        {
            doc_id
            for query_qrels in qrels.values()
            for doc_id in query_qrels
            if doc_id not in known_docs
        }
    )
    if missing_docs:
        preview = ", ".join(missing_docs[:5])
        raise ValueError(f"qrels 引用了语料中不存在的文档，共 {len(missing_docs)} 个，例如：{preview}")
    if not filtered_queries:
        raise ValueError("queries.jsonl 与 qrels 没有重合查询")
    return BeirDataset(documents=documents, queries=filtered_queries, qrels=qrels)


def load_external_candidates(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    candidates: dict[str, dict[str, str]] = {}
    for record in _read_jsonl(Path(path)):
        query_id = _pick(record, ("query_id", "_id", "id"))
        action = _pick(record, ("action", "operation"))
        text = _pick(record, ("text", "query", "rewrite"))
        candidates.setdefault(query_id, {})[action] = text
    return candidates
