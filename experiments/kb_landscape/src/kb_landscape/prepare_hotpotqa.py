from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 ReasonRAG HotpotQA dev 转为 BEIR 冒烟集")
    parser.add_argument("--input", required=True, help="ReasonRAG dev.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    return parser.parse_args()


def _doc_id(title: str, text: str) -> str:
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:20]
    return f"hotpot-{digest}"


def convert(input_path: str | Path, output_dir: str | Path, max_queries: int) -> dict:
    if max_queries <= 0:
        raise ValueError("max_queries 必须大于 0")
    documents: dict[str, dict[str, str]] = {}
    queries: list[dict[str, str]] = []
    qrels: list[tuple[str, str, int]] = []
    missing_supporting_titles: list[tuple[str, str]] = []

    with Path(input_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if len(queries) >= max_queries:
                break
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            query_id = str(row["id"])
            query_text = str(row["question"])
            metadata = row["metadata"]
            context = metadata["context"]
            titles = context["title"]
            sentences = context["sentences"]
            if len(titles) != len(sentences):
                raise ValueError(f"{query_id} 的 context.title 与 context.sentences 长度不同")

            current_title_to_ids: dict[str, list[str]] = {}
            for title, sentence_list in zip(titles, sentences):
                title_text = str(title)
                body = " ".join(str(sentence).strip() for sentence in sentence_list if str(sentence).strip())
                document_id = _doc_id(title_text, body)
                documents.setdefault(
                    document_id,
                    {"_id": document_id, "title": title_text, "text": body},
                )
                current_title_to_ids.setdefault(title_text, []).append(document_id)

            supporting = metadata["supporting_facts"]
            supporting_titles = list(dict.fromkeys(str(title) for title in supporting["title"]))
            for title in supporting_titles:
                document_ids = current_title_to_ids.get(title, [])
                if not document_ids:
                    missing_supporting_titles.append((query_id, title))
                    continue
                for document_id in document_ids:
                    qrels.append((query_id, document_id, 1))
            queries.append({"_id": query_id, "text": query_text})

    if missing_supporting_titles:
        preview = ", ".join(f"{query_id}:{title}" for query_id, title in missing_supporting_titles[:5])
        raise ValueError(f"supporting title 未出现在对应 context 中，共 {len(missing_supporting_titles)} 个：{preview}")
    if not queries:
        raise ValueError("没有读取到查询")

    output_path = Path(output_dir)
    qrels_dir = output_path / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    with (output_path / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents.values():
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    with (output_path / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")
    with (qrels_dir / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(qrels)

    summary = {
        "input": str(Path(input_path).resolve()),
        "output": str(output_path.resolve()),
        "queries": len(queries),
        "documents": len(documents),
        "qrels": len(qrels),
    }
    with (output_path / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = _parse_args()
    summary = convert(args.input, args.output_dir, args.max_queries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
