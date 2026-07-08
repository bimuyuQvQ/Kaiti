"""Build Elasticsearch BM25 index from ReasonRAG-style wiki18 jsonl corpus.

Memory safety notes:
- Python side streams jsonl line-by-line; it does NOT load the full corpus.
- Bulk insert uses elasticsearch.helpers.streaming_bulk with bounded chunk_size.
- Per-document `refresh=wait_for` is intentionally avoided (major ES memory/latency risk).
- The dominant OOM risk on shared servers is usually Elasticsearch JVM heap, not this script.
  Your local ES currently auto-sets heap to ~31GB; consider capping it before full indexing.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from typing import Generator, Iterable, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_contents(contents: str) -> Tuple[str, str]:
    if "\n" in contents:
        title_part, body = contents.split("\n", 1)
    else:
        title_part, body = contents, ""
    title = title_part.strip().strip('"')
    return title, body.strip()


def iter_jsonl_docs(
    path: str,
    max_docs: int = -1,
    start_doc: int = 0,
) -> Generator[dict, None, None]:
    """Stream one ES action per jsonl line; never materialize the whole corpus."""
    seen = 0
    indexed = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            if seen < start_doc:
                seen += 1
                continue

            obj = json.loads(line)
            doc_id = str(obj.get("id", seen))
            title, text = parse_contents(obj["contents"])
            yield {
                "_index": None,  # filled by streaming_bulk(index=...)
                "_id": doc_id,
                "_op_type": "index",
                "_source": {
                    "title": title,
                    "txt": text,
                },
            }
            seen += 1
            indexed += 1
            if max_docs > 0 and indexed >= max_docs:
                break


def bulk_index(
    es: Elasticsearch,
    index_name: str,
    actions: Iterable[dict],
    chunk_size: int,
    request_timeout: int,
) -> Tuple[int, int]:
    ok_count = 0
    err_count = 0
    progress = tqdm(unit="docs")
    for ok, _ in streaming_bulk(
        client=es,
        index=index_name,
        actions=actions,
        chunk_size=chunk_size,
        max_chunk_bytes=50 * 1024 * 1024,
        request_timeout=request_timeout,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            ok_count += 1
        else:
            err_count += 1
        progress.update(1)
    progress.close()
    return ok_count, err_count


def build_elasticsearch(
    corpus_pattern: str,
    index_name: str,
    max_docs: int = -1,
    start_doc: int = 0,
    skip_recreate: bool = False,
    chunk_size: int = 500,
    request_timeout: int = 120,
    hostname: str = "localhost",
) -> None:
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch

    corpus_files = sorted(glob.glob(corpus_pattern))
    if not corpus_files:
        raise FileNotFoundError(f"No corpus files matched: {corpus_pattern}")

    config = {
        "hostname": hostname,
        "index_name": index_name,
        "keys": {"title": "title", "body": "txt"},
        "timeout": request_timeout,
        "retry_on_timeout": True,
        "maxsize": 24,
        "number_of_shards": "default",
        "language": "english",
    }
    es_wrapper = ElasticSearch(config)
    es = es_wrapper.es

    if not skip_recreate:
        logger.info("Recreate index %s", index_name)
        es_wrapper.delete_index()
        time.sleep(5)
        es_wrapper.create_index()
    else:
        logger.info("Append to existing index %s (skip recreate)", index_name)

    def action_stream() -> Iterable[dict]:
        for corpus_file in corpus_files:
            logger.info("Indexing %s", corpus_file)
            yield from iter_jsonl_docs(
                corpus_file,
                max_docs=max_docs,
                start_doc=start_doc if corpus_file == corpus_files[0] else 0,
            )

    ok_count, err_count = bulk_index(
        es=es,
        index_name=index_name,
        actions=action_stream(),
        chunk_size=chunk_size,
        request_timeout=request_timeout,
    )
    logger.info("Bulk finished: ok=%s errors=%s", ok_count, err_count)
    if err_count:
        logger.warning("%s documents failed to index; check ES logs.", err_count)

    logger.info("Refreshing index %s once at the end", index_name)
    es.indices.refresh(index=index_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to wiki18 jsonl file or glob pattern",
    )
    parser.add_argument("--index_name", type=str, default="wiki")
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="Only index first N docs after start_doc; -1 means all",
    )
    parser.add_argument(
        "--start_doc",
        type=int,
        default=0,
        help="Skip first N docs in the jsonl (for resume)",
    )
    parser.add_argument(
        "--skip_recreate",
        action="store_true",
        help="Do not delete/recreate index before bulk insert",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Bulk batch size for streaming_bulk",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=120,
        help="Bulk request timeout in seconds",
    )
    parser.add_argument("--hostname", type=str, default="localhost")
    args = parser.parse_args()
    build_elasticsearch(
        args.corpus_path,
        index_name=args.index_name,
        max_docs=args.max_docs,
        start_doc=args.start_doc,
        skip_recreate=args.skip_recreate,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
        hostname=args.hostname,
    )
