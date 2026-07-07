"""
Step 2: Build FAISS index for extended corpus (OOM-safe version)

核心原则（严格执行）：
1) 绝不全量读取 JSONL（流式逐行）
2) 分块建索引并落盘（chunk_size=500000）
3) 每块完成后强制 del + gc.collect()
4) 最后通过磁盘级 merge_into 合并为完整索引

建议运行方式（先限内存，防止宿主机 OOM）：
  ulimit -v 83886080
  conda activate reasonrag
  python build_extend_index.py
"""

import gc
import json
import os
import argparse
from pathlib import Path
from typing import Generator

import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 减少 CUDA 内存碎片导致的偶发 OOM（需在 import torch 前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "indexes/wiki18_100w_extend.jsonl"
INDEX_OUT = BASE_DIR / "indexes/bge_Flat_wiki_extend.index"
PART_DIR = BASE_DIR / "indexes/extend_parts"

# ── 模型与构建参数 ─────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
START_BATCH_SIZE = 64
MIN_BATCH_SIZE = 8
CHUNK_SIZE = 500_000
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MERGE_BATCH_VECTORS = 100_000


def stream_contents(jsonl_path: Path) -> Generator[str, None, None]:
    """逐行流式读取 JSONL，yield 文本内容，绝不 readlines。"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj["contents"]


def encode_batch(texts: list[str], tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """BGE CLS pooling + L2 normalize，返回 float32 numpy。"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.detach().cpu().float().numpy()


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg


def encode_with_adaptive_batch(
    pending: list[str],
    remaining_in_chunk: int,
    current_batch_size: int,
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> tuple[np.ndarray, int, int]:
    """
    自适应 batch 编码：
    - 成功则返回 (emb, used_n, current_batch_size)
    - OOM 时自动减半 batch 重试，直到 MIN_BATCH_SIZE
    """
    local_batch_size = min(current_batch_size, len(pending), remaining_in_chunk)
    while True:
        use_n = min(local_batch_size, len(pending), remaining_in_chunk)
        if use_n <= 0:
            raise RuntimeError("encode_with_adaptive_batch got non-positive batch size.")
        texts = pending[:use_n]
        try:
            emb = encode_batch(texts, tokenizer, model)
            return emb, use_n, local_batch_size
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if local_batch_size <= MIN_BATCH_SIZE:
                raise RuntimeError(
                    f"CUDA OOM even at MIN_BATCH_SIZE={MIN_BATCH_SIZE}. "
                    "Try reducing MAX_LENGTH (e.g. 384/256) or use a larger GPU."
                ) from exc

            new_bs = max(MIN_BATCH_SIZE, local_batch_size // 2)
            print(
                f"\n[OOM] batch_size={local_batch_size} failed, "
                f"retry with batch_size={new_bs} ..."
            )
            local_batch_size = new_bs


def flush_current_chunk(
    chunk_index: faiss.Index,
    part_id: int,
    part_paths: list[Path],
) -> faiss.Index:
    """将当前 chunk 索引写盘，并彻底释放 chunk 对象。"""
    part_path = PART_DIR / f"index_part_{part_id}.index"
    print(f"\n[Chunk {part_id}] writing {chunk_index.ntotal:,} vectors -> {part_path.name}")
    faiss.write_index(chunk_index, str(part_path))
    part_paths.append(part_path)

    # 强制内存回收（按要求：del + gc.collect）
    del chunk_index
    gc.collect()

    return faiss.IndexFlatIP(768)


def merge_parts(part_paths: list[Path], out_path: Path) -> None:
    """
    磁盘级合并（兼容 IndexFlatIP）。
    注意：faiss.merge_into 仅适配 IVF 系列，这里改为 reconstruct_n + add。
    """
    if not part_paths:
        raise RuntimeError("No part index to merge.")

    print(f"\nLoading first part as merge target: {part_paths[0].name}")
    merged_index = faiss.read_index(str(part_paths[0]))
    print(f"  initial vectors: {merged_index.ntotal:,}")

    for i, part_path in enumerate(part_paths[1:], start=2):
        print(f"  [{i}/{len(part_paths)}] merging {part_path.name} ...")
        part_index = faiss.read_index(str(part_path))

        if part_index.d != merged_index.d:
            raise RuntimeError(
                f"Index dim mismatch: merged={merged_index.d}, part={part_index.d}, file={part_path}"
            )

        # 分批重建向量，避免单次占用过多内存
        total = part_index.ntotal
        for start in range(0, total, MERGE_BATCH_VECTORS):
            n = min(MERGE_BATCH_VECTORS, total - start)
            buf = np.empty((n, part_index.d), dtype=np.float32)
            part_index.reconstruct_n(start, n, buf)
            merged_index.add(buf)
            del buf
            gc.collect()

        del part_index
        gc.collect()

    print(f"\nSaving merged index -> {out_path}")
    faiss.write_index(merged_index, str(out_path))
    print(f"  merged total vectors: {merged_index.ntotal:,}")

    del merged_index
    gc.collect()


def main() -> None:
    global DEVICE, MAX_LENGTH, START_BATCH_SIZE, MIN_BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--start_batch_size", type=int, default=START_BATCH_SIZE)
    parser.add_argument("--min_batch_size", type=int, default=MIN_BATCH_SIZE)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    args = parser.parse_args()

    DEVICE = args.device
    MAX_LENGTH = args.max_length
    START_BATCH_SIZE = args.start_batch_size
    MIN_BATCH_SIZE = args.min_batch_size

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    PART_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        part_paths = sorted(
            PART_DIR.glob("index_part_*.index"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not part_paths:
            raise RuntimeError(f"No part files found in {PART_DIR}")
        print(f"Merge-only mode: found {len(part_paths)} parts in {PART_DIR}")
        merge_parts(part_paths=part_paths, out_path=INDEX_OUT)
        print(f"\nDone (merge only)! Final index: {INDEX_OUT}")
        return

    print("Loading encoder model...")
    print(f"  model: {MODEL_NAME}")
    print(f"  device: {DEVICE}")
    print(
        f"  start_batch_size: {START_BATCH_SIZE}, "
        f"min_batch_size: {MIN_BATCH_SIZE}, chunk_size: {CHUNK_SIZE}"
    )
    if DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        # 显式绑定目标 GPU，避免默认设备漂移
        if ":" in DEVICE:
            torch.cuda.set_device(int(DEVICE.split(":")[1]))
        else:
            torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=args.local_files_only)
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
            local_files_only=args.local_files_only,
        ).to(DEVICE)
    except Exception as exc:
        if is_cuda_oom_error(exc):
            raise RuntimeError(
                "CUDA OOM during model load. "
                "Please check target GPU usage (nvidia-smi), or switch device with "
                "`--device cuda:1` etc. "
                "If you set `ulimit -v 83886080`, it may be too strict for CUDA driver "
                "virtual memory mapping in your shell; try a larger limit."
            ) from exc
        raise
    model.eval()

    print(f"\nStreaming corpus from: {CORPUS_PATH}")
    stream = stream_contents(CORPUS_PATH)

    part_paths: list[Path] = []
    part_id = 0
    chunk_doc_count = 0
    total_doc_count = 0
    pending: list[str] = []
    current_batch_size = START_BATCH_SIZE
    chunk_index = faiss.IndexFlatIP(768)

    pbar = tqdm(desc="indexing (stream)", unit="doc")
    for text in stream:
        pending.append(text)

        while len(pending) >= current_batch_size:
            remaining = CHUNK_SIZE - chunk_doc_count
            if remaining == 0:
                chunk_index = flush_current_chunk(chunk_index, part_id, part_paths)
                part_id += 1
                chunk_doc_count = 0
                remaining = CHUNK_SIZE

            emb, used_n, bs_after = encode_with_adaptive_batch(
                pending=pending,
                remaining_in_chunk=remaining,
                current_batch_size=current_batch_size,
                tokenizer=tokenizer,
                model=model,
            )
            pending = pending[used_n:]
            chunk_index.add(emb)

            batch_n = used_n
            current_batch_size = bs_after
            chunk_doc_count += batch_n
            total_doc_count += batch_n
            pbar.update(batch_n)

            # 强制释放 batch 张量/数组
            del emb
            gc.collect()

    # 处理尾部不足 batch 的文本
    while pending:
        remaining = CHUNK_SIZE - chunk_doc_count
        if remaining == 0:
            chunk_index = flush_current_chunk(chunk_index, part_id, part_paths)
            part_id += 1
            chunk_doc_count = 0
            remaining = CHUNK_SIZE

        emb, used_n, bs_after = encode_with_adaptive_batch(
            pending=pending,
            remaining_in_chunk=remaining,
            current_batch_size=current_batch_size,
            tokenizer=tokenizer,
            model=model,
        )
        pending = pending[used_n:]
        chunk_index.add(emb)

        batch_n = used_n
        current_batch_size = bs_after
        chunk_doc_count += batch_n
        total_doc_count += batch_n
        pbar.update(batch_n)

        del emb
        gc.collect()

    pbar.close()
    print(f"\nTotal encoded docs: {total_doc_count:,}")

    if chunk_index.ntotal > 0:
        chunk_index = flush_current_chunk(chunk_index, part_id, part_paths)
        part_id += 1
    else:
        del chunk_index
        gc.collect()

    if not part_paths:
        raise RuntimeError("No part index generated. Check corpus format.")

    # 编码阶段结束后释放模型
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nGenerated {len(part_paths)} part indexes in: {PART_DIR}")

    # 磁盘级合并
    merge_parts(part_paths=part_paths, out_path=INDEX_OUT)

    print(
        "\nDone!\n"
        f"Final index: {INDEX_OUT}\n"
        "Reminder: run with memory guard before execution:\n"
        "  ulimit -v 83886080"
    )


if __name__ == "__main__":
    main()
