import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import BIOASQ, IIRC, HotpotQA, PubmedQA, StrategyQA, WikiMultiHopQA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSWER_MARKER_RE = re.compile(r"\bthe answer is\b", re.IGNORECASE)
REPEATED_ANSWER_RE = re.compile(r"\b(?:so\s+)?the answer is\b", re.IGNORECASE)
QUESTION_TAIL_RE = re.compile(r"(?:(?<=\s)|(?<=\.))Question\b", re.IGNORECASE)
SPLIT_WORDS = ["Question:", "#10000000", "Note:"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument(
        "--extract_mode",
        choices=["original", "first_answer_span"],
        default="first_answer_span",
    )
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r", encoding="utf-8") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    args.extract_mode = tmp.extract_mode
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
    return data


def strip_generation_tail(text):
    for word in SPLIT_WORDS:
        pos = text.find(word)
        if pos != -1 and pos > 0:
            text = text[:pos]
    return text


def extract_first_answer_span(text):
    text = strip_generation_tail(text)
    match = ANSWER_MARKER_RE.search(text)
    if not match:
        return ""
    text = text[match.end():].strip()

    repeated = REPEATED_ANSWER_RE.search(text)
    if repeated:
        text = text[:repeated.start()].strip()
    question_tail = QUESTION_TAIL_RE.search(text)
    if question_tail:
        text = text[:question_tail.start()].strip()

    for suffix in ["</s>", "<|endoftext|>"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def main():
    args = get_args()
    logger.info(args)
    data = load_dataset(args)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"],
            t["answer_id"] if "answer_id" in t else None,
        ]

    metrics = ["EM", "F1", "Precision", "Recall", "Accuracy"]
    count_list = []
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
    value = [[] for _ in range(len(metrics))]

    with open(os.path.join(args.output_dir, "output.txt"), "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    detail_path = os.path.join(args.output_dir, f"details_{args.extract_mode}.txt")
    pred_out = open(detail_path, "w", encoding="utf-8")
    total_correct = 0

    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        ground_truth, ground_truth_id = dataset[qid]

        if args.extract_mode == "original":
            pred = data.get_real_prediction(strip_generation_tail(pred))
        else:
            pred = extract_first_answer_span(pred)

        em_ret = data.exact_match_score(pred, ground_truth, ground_truth_id)
        f1_ret = data.f1_score(pred, ground_truth, ground_truth_id)

        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i + 1].append(f1_ret[k])
        if em_ret["correct"]:
            total_correct += 1

        for i, k in enumerate(count_list):
            value[i + 5].append(rd[k])

        detail = {
            "qid": qid,
            "final_pred": pred,
            "EM": str(em_ret["correct"]),
            "F1": str(f1_ret["f1"]),
        }
        if "sample_index" in rd:
            detail["sample_index"] = rd["sample_index"]
        pred_out.write(json.dumps(detail, ensure_ascii=False) + "\n")
    pred_out.close()

    acc = total_correct / len(lines)
    ret = []
    for i, metric in enumerate(metrics):
        if metric == "Accuracy":
            ret.append([metric, acc])
        else:
            ret.append([metric, np.array(value[i]).mean()])
    result_path = os.path.join(args.output_dir, f"result_{args.extract_mode}.tsv")
    pd.DataFrame(ret).to_csv(result_path, index=False, header=False)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
