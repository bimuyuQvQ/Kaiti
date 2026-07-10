import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    procs = []
    for shard_index in range(args.num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard_index)
        cmd = [
            args.python,
            "main_shard.py",
            "--config_path",
            args.config_path,
            "--run_dir",
            args.run_dir,
            "--shard_index",
            str(shard_index),
            "--num_shards",
            str(args.num_gpus),
        ]
        log_path = os.path.join(args.run_dir, f"shard_{shard_index}.log")
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((shard_index, log_file, proc))

    failed = []
    for shard_index, log_file, proc in procs:
        code = proc.wait()
        log_file.close()
        if code != 0:
            failed.append((shard_index, code))

    if failed:
        raise SystemExit(f"failed shards: {failed}")

    subprocess.check_call([
        args.python,
        "merge_shards.py",
        "--run_dir",
        args.run_dir,
        "--num_shards",
        str(args.num_gpus),
    ])


if __name__ == "__main__":
    main()
