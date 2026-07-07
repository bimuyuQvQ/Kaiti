#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(name, values, thresholds):
    values = list(values)
    values_sorted = sorted(values)
    n = len(values_sorted)
    if n == 0:
        return {"name": name, "count": 0}
    summary = {
        "name": name,
        "count": n,
        "min": int(values_sorted[0]),
        "mean": float(mean(values_sorted)),
        "p50": float(percentile(values_sorted, 50)),
        "p90": float(percentile(values_sorted, 90)),
        "p95": float(percentile(values_sorted, 95)),
        "p99": float(percentile(values_sorted, 99)),
        "max": int(values_sorted[-1]),
    }
    over = {}
    for t in thresholds:
        c = sum(1 for v in values_sorted if v > t)
        over[f">{t}"] = {"count": c, "ratio": c / n}
    summary["over_threshold"] = over
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze token length distribution for RAG_ProGuide.")
    parser.add_argument(
        "--dataset",
        default="/data1/home/lmy/LLaMA-Factory/data/RAG_ProGuide.json",
        help="Path to dataset json file.",
    )
    parser.add_argument(
        "--tokenizer",
        default="/data1/home/lmy/models/Qwen2.5-7B-Instruct",
        help="Tokenizer/model path.",
    )
    parser.add_argument(
        "--thresholds",
        default="512,1024,1536,2048,3072,4096,8192,10000",
        help="Comma-separated thresholds.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save full JSON report.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    thresholds = [int(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompt_len = []
    chosen_len = []
    rejected_len = []
    prompt_chosen_len = []
    prompt_rejected_len = []

    for ex in data:
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")

        p = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        c = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        r = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
        pc = len(tokenizer(prompt + "\n" + chosen, add_special_tokens=False)["input_ids"])
        pr = len(tokenizer(prompt + "\n" + rejected, add_special_tokens=False)["input_ids"])

        prompt_len.append(p)
        chosen_len.append(c)
        rejected_len.append(r)
        prompt_chosen_len.append(pc)
        prompt_rejected_len.append(pr)

    max_pair_len = [max(a, b) for a, b in zip(prompt_chosen_len, prompt_rejected_len)]

    report = {
        "dataset": str(dataset_path),
        "num_examples": len(data),
        "tokenizer": args.tokenizer,
        "thresholds": thresholds,
        "stats": [
            summarize("prompt", prompt_len, thresholds),
            summarize("chosen", chosen_len, thresholds),
            summarize("rejected", rejected_len, thresholds),
            summarize("prompt+chosen", prompt_chosen_len, thresholds),
            summarize("prompt+rejected", prompt_rejected_len, thresholds),
            summarize("max(prompt+chosen, prompt+rejected)", max_pair_len, thresholds),
        ],
    }

    print("=" * 88)
    print("RAG_ProGuide length analysis (tokenized by Qwen tokenizer)")
    print("=" * 88)
    print(f"Dataset:    {report['dataset']}")
    print(f"Examples:   {report['num_examples']}")
    print(f"Tokenizer:  {report['tokenizer']}")
    print(f"Thresholds: {thresholds}")
    print("-" * 88)
    for s in report["stats"]:
        print(
            f"[{s['name']}] "
            f"min={s['min']} mean={s['mean']:.1f} p50={s['p50']:.1f} "
            f"p90={s['p90']:.1f} p95={s['p95']:.1f} p99={s['p99']:.1f} max={s['max']}"
        )
        over_str = ", ".join(
            [f"{k}:{v['count']} ({v['ratio']*100:.2f}%)" for k, v in s["over_threshold"].items()]
        )
        print(f"  over: {over_str}")
    print("-" * 88)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report to: {out}")


if __name__ == "__main__":
    main()
