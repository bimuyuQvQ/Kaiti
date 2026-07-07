"""
Step 1: Build wiki18_100w_extend.jsonl
从 HotpotQA / 2WikiMultiHopQA dev 集的 context 字段提取 golden documents，
按 ~100 词切块，与 wiki18_100w.jsonl 格式对齐，去重后追加，生成 _extend 版本。
同时输出 new_docs_only.jsonl 供增量索引建立使用。

用法：
  python build_extend_corpus.py

输出：
  indexes/wiki18_100w_extend.jsonl
  indexes/new_docs_only.jsonl      （仅新增条目，用于 build_extend_index.py）
"""

import json
import ast
import hashlib
from pathlib import Path
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
CORPUS_IN    = BASE_DIR / "indexes/wiki18_100w.jsonl"
CORPUS_OUT   = BASE_DIR / "indexes/wiki18_100w_extend.jsonl"
NEW_DOCS_OUT = BASE_DIR / "indexes/new_docs_only.jsonl"

DATASETS = [
    BASE_DIR / "dataset/hotpotqa/dev.jsonl",
    BASE_DIR / "dataset/2wikimultihopqa/dev.jsonl",
]

CHUNK_WORDS = 100   # 对齐 wiki18_100w 的切分粒度

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def parse_metadata(raw) -> dict:
    """兼容 metadata 为 dict / Python repr 字符串 / JSON 字符串。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            try:
                return json.loads(raw)
            except Exception:
                return {}
    return {}


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    """按 chunk_words 切块；最后一块不足时保留。"""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_words])
        for i in range(0, len(words), chunk_words)
        if words[i : i + chunk_words]
    ]


def make_contents(title: str, text: str) -> str:
    """与 wiki18_100w.jsonl 格式一致：第一行是 "Title"，第二行起是正文。"""
    return f'"{title}"\n{text}'


def text_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ── Step 1：收集现有语料的内容 hash（用于去重） ────────────────────────────────
print("Loading existing corpus hashes (this may take ~2 min for 21M lines)...")
existing_hashes: set[str] = set()
with CORPUS_IN.open() as f:
    for line in tqdm(f, desc="hashing wiki18_100w"):
        d = json.loads(line)
        existing_hashes.add(text_hash(d["contents"].strip()))

print(f"  Existing entries: {len(existing_hashes):,}")

# ── Step 2：从 QA 数据集提取 context 文章 ─────────────────────────────────────
new_docs: list[dict] = []
seen_new: set[str] = set()

for ds_path in DATASETS:
    if not ds_path.exists():
        print(f"  Skipping {ds_path} (not found)")
        continue

    ds_name = ds_path.parent.name
    print(f"\nProcessing {ds_name} ({ds_path.name})...")

    with ds_path.open() as f:
        for line in tqdm(f, desc=ds_name):
            item = json.loads(line)
            meta = parse_metadata(item.get("metadata", {}))
            ctx  = meta.get("context", {})

            titles    = ctx.get("title", [])
            # hotpotqa 用 "sentences"，2wikimultihopqa 用 "content"
            sents_key = "sentences" if "sentences" in ctx else "content"
            sents_all = ctx.get(sents_key, [])

            for title, sents in zip(titles, sents_all):
                # 合并该文章所有句子为一段文本
                full_text = " ".join(s.strip() for s in sents if isinstance(s, str) and s.strip())
                if not full_text:
                    continue

                # 切块对齐 100 words
                for chunk in chunk_text(full_text, CHUNK_WORDS):
                    contents = make_contents(title, chunk)
                    h = text_hash(contents.strip())
                    if h in existing_hashes or h in seen_new:
                        continue
                    seen_new.add(h)
                    new_docs.append({"id": f"extend_{len(new_docs)}", "contents": contents})

print(f"\nNew unique documents to add: {len(new_docs):,}")

# ── Step 3：追加写入 _extend 版本，并输出仅新增文件 ───────────────────────────
print(f"\nWriting {CORPUS_OUT.name} ...")
import shutil
shutil.copy(CORPUS_IN, CORPUS_OUT)

with CORPUS_OUT.open("a") as fout, NEW_DOCS_OUT.open("w") as fnew:
    for doc in tqdm(new_docs, desc="appending"):
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        fout.write(line)
        fnew.write(line)

print(f"\nDone.")
print(f"  {CORPUS_OUT}  ({len(existing_hashes) + len(new_docs):,} total entries)")
print(f"  {NEW_DOCS_OUT}  ({len(new_docs):,} new entries)")
print(f"\nNext step: python build_extend_index.py")
