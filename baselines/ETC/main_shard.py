import argparse
import gc
import json
import logging
import os
from copy import copy

import torch
from tqdm import tqdm

from data import BIOASQ, IIRC, HotpotQA, PubmedQA, StrategyQA, WikiMultiHopQA
from generate import ETC


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--oom_retries", type=int, default=1)
    cli_args = parser.parse_args()

    with open(cli_args.config_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = cli_args.config_path
    args.output_dir = os.path.join(cli_args.run_dir, f"shard_{cli_args.shard_index}")
    args.run_dir = cli_args.run_dir
    args.shard_index = cli_args.shard_index
    args.num_shards = cli_args.num_shards
    args.oom_retries = cli_args.oom_retries
    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return args


def load_dataset(args):
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    elif args.dataset == "bioasq_7b_yesno":
        data = BIOASQ(args.data_path)
    elif args.dataset == "pubmedQA":
        data = PubmedQA(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
    return data


def load_completed_indices(output_path):
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"输出文件第 {line_number} 行不是完整 JSON: {output_path}") from exc
            sample_index = row.get("sample_index")
            if sample_index is None:
                raise ValueError(f"输出文件第 {line_number} 行缺少 sample_index: {output_path}")
            if sample_index in completed:
                raise ValueError(f"输出文件存在重复 sample_index={sample_index}: {output_path}")
            completed.add(sample_index)
    return completed


def validate_existing_config(args, config_path):
    if not os.path.exists(config_path):
        return
    with open(args.config_path, "r", encoding="utf-8") as f:
        requested_config = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        existing_config = json.load(f)
    ignored_keys = {"output_dir", "config_path"}
    mismatches = {
        key: {"existing": existing_config.get(key), "requested": value}
        for key, value in requested_config.items()
        if key not in ignored_keys
        and key in existing_config
        and existing_config[key] != value
    }
    if mismatches:
        raise ValueError(f"续跑配置与已有结果不一致: {mismatches}")


def main():
    args = get_args()
    logger.info(args)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_config_path = os.path.join(args.output_dir, "config.json")
    validate_existing_config(args, shard_config_path)
    data = load_dataset(args)
    args.effective_sample_count = len(data)
    with open(shard_config_path, "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)
    sample_indices = list(range(args.shard_index, len(data), args.num_shards))
    shard_data = data.select(sample_indices)

    output_path = os.path.join(args.output_dir, "output.txt")
    completed_indices = load_completed_indices(output_path)
    pending_local_indices = [
        local_i
        for local_i, sample_index in enumerate(sample_indices)
        if sample_index not in completed_indices
    ]
    unexpected_indices = completed_indices.difference(sample_indices)
    if unexpected_indices:
        raise ValueError(f"输出文件包含不属于当前分片的 sample_index: {sorted(unexpected_indices)}")
    if not pending_local_indices:
        logger.info("shard %s 已完成，无需加载模型", args.shard_index)
        return

    model = ETC(args)
    error_path = os.path.join(args.output_dir, "errors.jsonl")
    logger.info(
        "start shard inference: %s/%s, total=%s, completed=%s, pending=%s",
        args.shard_index,
        args.num_shards,
        len(shard_data),
        len(completed_indices),
        len(pending_local_indices),
    )
    with open(output_path, "a", encoding="utf-8") as output_file:
        for local_i in tqdm(pending_local_indices, initial=len(completed_indices), total=len(shard_data)):
            last_counter = copy(model.counter)
            entry = shard_data[local_i]
            for attempt in range(args.oom_retries + 1):
                try:
                    pred = model.inference(entry["question"], entry["demo"], entry["case"]).strip()
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    model.counter = copy(last_counter)
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.exception(
                        "sample_index=%s 发生 CUDA OOM，attempt=%s/%s",
                        sample_indices[local_i],
                        attempt + 1,
                        args.oom_retries + 1,
                    )
                    if attempt >= args.oom_retries:
                        with open(error_path, "a", encoding="utf-8") as error_file:
                            error_file.write(json.dumps({
                                "qid": entry["qid"],
                                "sample_index": sample_indices[local_i],
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }, ensure_ascii=False) + "\n")
                            error_file.flush()
                        raise
            ret = {
                "qid": entry["qid"],
                "sample_index": sample_indices[local_i],
                "prediction": pred,
            }
            if args.use_counter:
                ret.update(model.counter.calc(last_counter))
            output_file.write(json.dumps(ret, ensure_ascii=False) + "\n")
            output_file.flush()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
