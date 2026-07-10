import argparse
import datetime
import os
import signal
import subprocess
import sys
import time


def stop_handler(_signum, _frame):
    raise KeyboardInterrupt


def start_shard(args, shard_index, restart_count, gpu_ids):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
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
        "--oom_retries",
        str(args.oom_retries),
    ]
    log_path = os.path.join(args.run_dir, f"shard_{shard_index}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    log_file.write(
        f"\n===== start shard {shard_index}, restart {restart_count}, {started_at} =====\n"
    )
    log_file.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"started shard {shard_index} on GPUs {gpu_ids}, pid={proc.pid}", flush=True)
    return log_file, proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpus_per_worker", type=int, default=2)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--max_restarts", type=int, default=2)
    parser.add_argument("--oom_retries", type=int, default=1)
    args = parser.parse_args()
    if args.gpus_per_worker <= 0 or args.num_gpus % args.gpus_per_worker != 0:
        raise ValueError("num_gpus 必须能被 gpus_per_worker 整除")
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    os.makedirs(args.run_dir, exist_ok=True)
    restart_counts = {shard_index: 0 for shard_index in range(args.num_gpus)}
    exhausted = []
    pending = list(range(args.num_gpus))
    free_gpu_slots = [
        ",".join(str(gpu_index) for gpu_index in range(slot_start, slot_start + args.gpus_per_worker))
        for slot_start in range(0, args.num_gpus, args.gpus_per_worker)
    ]
    active = {}

    def fill_free_slots():
        while pending and free_gpu_slots:
            shard_index = pending.pop(0)
            gpu_ids = free_gpu_slots.pop(0)
            log_file, proc = start_shard(
                args,
                shard_index,
                restart_counts[shard_index],
                gpu_ids,
            )
            active[shard_index] = (gpu_ids, log_file, proc)

    fill_free_slots()
    try:
        while active:
            for shard_index, (gpu_ids, log_file, proc) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log_file.close()
                del active[shard_index]
                if code == 0:
                    print(f"shard {shard_index} completed", flush=True)
                    free_gpu_slots.append(gpu_ids)
                    continue
                restart_counts[shard_index] += 1
                if restart_counts[shard_index] <= args.max_restarts:
                    print(
                        f"shard {shard_index} failed with code {code}; "
                        f"restarting from checkpoint ({restart_counts[shard_index]}/{args.max_restarts})",
                        flush=True,
                    )
                    log_file, proc = start_shard(
                        args,
                        shard_index,
                        restart_counts[shard_index],
                        gpu_ids,
                    )
                    active[shard_index] = (gpu_ids, log_file, proc)
                else:
                    exhausted.append((shard_index, code))
                    free_gpu_slots.append(gpu_ids)
            fill_free_slots()
            if active:
                time.sleep(1)
    except KeyboardInterrupt:
        print("stopping active shards", flush=True)
        for _, log_file, proc in active.values():
            proc.terminate()
            log_file.close()
        for _, _, proc in active.values():
            proc.wait()
        raise

    if exhausted:
        raise SystemExit(f"failed shards after retries: {exhausted}")

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
