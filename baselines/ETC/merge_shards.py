import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    rows = []
    report = []
    seen_indices = set()
    for shard_index in range(args.num_shards):
        shard_dir = os.path.join(args.run_dir, f"shard_{shard_index}")
        output_path = os.path.join(shard_dir, "output.txt")
        if not os.path.exists(output_path):
            raise FileNotFoundError(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            shard_rows = [json.loads(line) for line in f if line.strip()]
        shard_indices = [row.get("sample_index") for row in shard_rows]
        if any(sample_index is None for sample_index in shard_indices):
            raise ValueError(f"shard {shard_index} 存在缺少 sample_index 的结果")
        if len(shard_indices) != len(set(shard_indices)):
            raise ValueError(f"shard {shard_index} 存在重复 sample_index")
        wrong_indices = [
            sample_index
            for sample_index in shard_indices
            if sample_index % args.num_shards != shard_index
        ]
        if wrong_indices:
            raise ValueError(f"shard {shard_index} 包含错误分片索引: {wrong_indices[:10]}")
        duplicate_indices = seen_indices.intersection(shard_indices)
        if duplicate_indices:
            raise ValueError(f"跨分片存在重复 sample_index: {sorted(duplicate_indices)[:10]}")
        seen_indices.update(shard_indices)
        rows.extend(shard_rows)
        report.append({"shard_index": shard_index, "count": len(shard_rows)})

    if rows and "sample_index" in rows[0]:
        rows.sort(key=lambda row: row["sample_index"])

    shard0_config = os.path.join(args.run_dir, "shard_0", "config.json")
    config = None
    if os.path.exists(shard0_config):
        with open(shard0_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        expected_total = config.get("effective_sample_count", config.get("sample"))
        if isinstance(expected_total, int) and expected_total >= 0:
            expected_indices = set(range(expected_total))
            missing_indices = expected_indices.difference(seen_indices)
            extra_indices = seen_indices.difference(expected_indices)
            if missing_indices or extra_indices:
                raise ValueError(
                    "分片结果不完整: "
                    f"missing={sorted(missing_indices)[:20]}, "
                    f"extra={sorted(extra_indices)[:20]}"
                )

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "output.txt"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if config is not None:
        config["output_dir"] = args.run_dir
        with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    with open(os.path.join(args.run_dir, "merge_report.json"), "w", encoding="utf-8") as f:
        json.dump({"total": len(rows), "shards": report}, f, indent=4)

    print(f"merged {len(rows)} rows into {os.path.join(args.run_dir, 'output.txt')}")


if __name__ == "__main__":
    main()
