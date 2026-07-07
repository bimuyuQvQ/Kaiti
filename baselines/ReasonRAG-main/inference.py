import argparse
import os
import sys
import glob
import ctypes

# vLLM 0.23 需要 cu13 runtime；必须在 import vllm 前写入环境变量（子进程也会继承）
_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
_site_pkg = os.path.join(_prefix, "lib/python3.10/site-packages")
_candidate_lib_dirs = [
    os.path.join(_site_pkg, "nvidia/cu13/lib"),
    os.path.join(_site_pkg, "nvidia/cuda_runtime/lib"),
]
# also include any nvidia/*/lib for robustness across package layouts
_candidate_lib_dirs.extend(glob.glob(os.path.join(_site_pkg, "nvidia/*/lib")))
_candidate_lib_dirs = [d for d in dict.fromkeys(_candidate_lib_dirs) if os.path.isdir(d)]

if _candidate_lib_dirs:
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_candidate_lib_dirs) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

# proactively preload libcudart to avoid vllm import failure in some shells
for _lib_dir in _candidate_lib_dirs:
    _cudart13 = os.path.join(_lib_dir, "libcudart.so.13")
    if os.path.isfile(_cudart13):
        try:
            ctypes.CDLL(_cudart13, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass
        break

import yaml
from flashrag.config import Config
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--max_iter", default=8, type=int)
parser.add_argument("--retrieval_top_k", default=3, type=int)
parser.add_argument("--gpu_id", default="0", type=str, help='GPU id, e.g. "0" or "0,1"')
parser.add_argument("--run_tag", default="", type=str, help="Optional tag to distinguish parallel runs")
parser.add_argument("--lora_path", default=None, type=str, help="Optional LoRA adapter dir; enables vLLM LoRA on base --model")
parser.add_argument("--index_path", default="indexes/bge_Flat.index", type=str)
parser.add_argument("--corpus_path", default="indexes/wiki18_100w.jsonl", type=str)
args = parser.parse_args()

root_dir = 'output'

def load_config_from_yaml(yaml_file):
    try:
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return {}

default_config = load_config_from_yaml("my_config.yaml")

config_dict = {
    "data_dir": "dataset/",
    "dataset_name": args.dataset_name,
    "index_path": args.index_path,
    "retrieval_method": "bge",
    "corpus_path": args.corpus_path,
    "model2path": {
        "bge": "BAAI/bge-base-en-v1.5",
        "e5": "intfloat/e5-base-v2",
        "qwen2.5": "Qwen/Qwen2.5-7B",
        "qwen2.5-instruct": "Qwen/Qwen2.5-7B-Instruct",
    },
    "generator_model": args.model,
    "generator_batch_size": 1,
    "framework": "vllm",
    "gpu_id": args.gpu_id,
    "faiss_gpu": False,
    "retrieval_batch_size": 256,
    "gpu_memory_utilization": 0.85,
    "metrics": ["em", "f1", "acc", "recall", "precision"],
    "retrieval_topk": args.retrieval_top_k,
    "save_intermediate_data": True,
    "save_note": (
        f"{os.path.basename(args.model.rstrip('/'))}"
        f"{'_lora_' + os.path.basename(args.lora_path.rstrip('/')) if args.lora_path else ''}"
        f"_{args.dataset_name}_iter{args.max_iter}_topk{args.retrieval_top_k}_gpu{args.gpu_id.replace(',', '-')}"
        f"{'_' + args.run_tag if args.run_tag else ''}"
    ),
}

if args.lora_path:
    config_dict["generator_lora_path"] = args.lora_path

answer_format = "answer"
max_iter = 10

config_dict = {**default_config, **config_dict}
config = Config(config_dict=config_dict)

dataset_path = config["dataset_path"]
split_path = os.path.join(dataset_path, "test.jsonl")
data_split = "test"
if not os.path.exists(split_path):
    if os.path.exists(os.path.join(dataset_path, "dev.jsonl")):
        data_split = "dev"
    elif os.path.exists(os.path.join(dataset_path, "val.jsonl")):
        data_split = "val"
    else:
        data_split = "None"

all_split = get_dataset(config)
test_data = all_split[data_split]

pipeline = ReasonRAGPipeline(config, prompt_template=None, answer_format=answer_format, max_iter=args.max_iter, max_children=2, max_rollouts=64)
output_dataset = pipeline.run(test_data, batch_size=1000, do_eval=True)
