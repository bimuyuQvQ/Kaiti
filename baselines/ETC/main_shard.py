import argparse
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
    cli_args = parser.parse_args()

    with open(cli_args.config_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = cli_args.config_path
    args.output_dir = os.path.join(cli_args.run_dir, f"shard_{cli_args.shard_index}")
    args.run_dir = cli_args.run_dir
    args.shard_index = cli_args.shard_index
    args.num_shards = cli_args.num_shards
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


def main():
    args = get_args()
    logger.info(args)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=4)

    data = load_dataset(args)
    sample_indices = list(range(args.shard_index, len(data), args.num_shards))
    shard_data = data.select(sample_indices)

    model = ETC(args)
    output_path = os.path.join(args.output_dir, "output.txt")
    logger.info("start shard inference: %s/%s, %s samples", args.shard_index, args.num_shards, len(shard_data))
    with open(output_path, "w", encoding="utf-8") as output_file:
        for local_i in tqdm(range(len(shard_data))):
            last_counter = copy(model.counter)
            entry = shard_data[local_i]
            pred = model.inference(entry["question"], entry["demo"], entry["case"]).strip()
            ret = {
                "qid": entry["qid"],
                "sample_index": sample_indices[local_i],
                "prediction": pred,
            }
            if args.use_counter:
                ret.update(model.counter.calc(last_counter))
            output_file.write(json.dumps(ret, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
