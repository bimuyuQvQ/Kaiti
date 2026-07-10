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
    for shard_index in range(args.num_shards):
        shard_dir = os.path.join(args.run_dir, f"shard_{shard_index}")
        output_path = os.path.join(shard_dir, "output.txt")
        if not os.path.exists(output_path):
            raise FileNotFoundError(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            shard_rows = [json.loads(line) for line in f]
        rows.extend(shard_rows)
        report.append({"shard_index": shard_index, "count": len(shard_rows)})

    if rows and "sample_index" in rows[0]:
        rows.sort(key=lambda row: row["sample_index"])

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "output.txt"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    shard0_config = os.path.join(args.run_dir, "shard_0", "config.json")
    if os.path.exists(shard0_config):
        with open(shard0_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["output_dir"] = args.run_dir
        with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    with open(os.path.join(args.run_dir, "merge_report.json"), "w", encoding="utf-8") as f:
        json.dump({"total": len(rows), "shards": report}, f, indent=4)

    print(f"merged {len(rows)} rows into {os.path.join(args.run_dir, 'output.txt')}")


if __name__ == "__main__":
    main()
