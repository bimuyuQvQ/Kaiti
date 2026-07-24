"""服务器实验环境的可复现检查。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any


GATE_PACKAGES = ("torch", "datasets", "transformers", "pyarrow", "yaml", "rank_bm25")
SFT_PACKAGES = ("accelerate", "bitsandbytes", "peft", "sentence_transformers")


def package_version(import_name: str) -> dict[str, Any]:
    distribution_names = {
        "yaml": "PyYAML",
        "rank_bm25": "rank-bm25",
        "sentence_transformers": "sentence-transformers",
    }
    try:
        version = importlib.metadata.version(distribution_names.get(import_name, import_name))
        return {"available": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"available": False, "version": None}


def inspect_model_cache(cache_root: Path, model_id: str) -> dict[str, Any]:
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    snapshot_dirs = sorted(path for path in snapshots.glob("*") if path.is_dir())
    useful_files: list[str] = []
    for snapshot in snapshot_dirs:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if (snapshot / name).exists():
                useful_files.append(str((snapshot / name).relative_to(model_dir)))
    return {
        "model_id": model_id,
        "cache_dir": str(model_dir),
        "exists": model_dir.exists(),
        "snapshot_count": len(snapshot_dirs),
        "metadata_files": useful_files,
    }


def inspect_cuda(matrix_size: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "hard_failure": f"无法导入 torch: {exc}"}

    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if not result["available"]:
        result["hard_failure"] = "torch.cuda.is_available() 为 False"
        return result

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    try:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        dtype = torch.bfloat16 if result["bf16_supported"] else torch.float16
        start = time.perf_counter()
        left = torch.randn((matrix_size, matrix_size), device="cuda", dtype=dtype)
        right = torch.randn((matrix_size, matrix_size), device="cuda", dtype=dtype)
        product = left @ right
        checksum = float(product[0, 0].item())
        torch.cuda.synchronize()
        result["smoke_test"] = {
            "passed": True,
            "matrix_size": matrix_size,
            "dtype": str(dtype),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
            "checksum": checksum,
        }
        del left, right, product
        torch.cuda.empty_cache()
    except Exception as exc:  # GPU/driver errors must be preserved verbatim.
        result["smoke_test"] = {"passed": False, "error": repr(exc)}
        result["hard_failure"] = "CUDA 矩阵乘烟雾测试失败"
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    disk = shutil.disk_usage(args.disk_path)
    packages = {
        name: package_version(name)
        for name in sorted(set(GATE_PACKAGES + SFT_PACKAGES))
    }
    gate_missing = [name for name in GATE_PACKAGES if not packages[name]["available"]]
    sft_missing = [name for name in SFT_PACKAGES if not packages[name]["available"]]
    cuda = inspect_cuda(args.matrix_size)
    models = [
        inspect_model_cache(args.hf_cache, model_id)
        for model_id in args.required_model
    ]
    hard_failures = []
    if gate_missing:
        hard_failures.append(f"门槛实验缺少依赖: {', '.join(gate_missing)}")
    if cuda.get("hard_failure"):
        hard_failures.append(cuda["hard_failure"])
    for model in models:
        if not model["exists"] or not model["snapshot_count"]:
            hard_failures.append(f"模型缓存不完整: {model['model_id']}")

    return {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "disk": {
            "path": str(args.disk_path),
            "free_gib": round(disk.free / 1024**3, 2),
            "total_gib": round(disk.total / 1024**3, 2),
        },
        "packages": packages,
        "gate_missing_packages": gate_missing,
        "sft_missing_packages": sft_missing,
        "cuda": cuda,
        "models": models,
        "hard_failures": hard_failures,
        "ready_for_gate": not hard_failures,
        "ready_for_sft": not hard_failures and not sft_missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, default=Path("/data1"))
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument(
        "--required-model",
        action="append",
        default=[
            "BAAI/bge-base-en-v1.5",
            "reasonrag/Qwen2.5-7B-Instruct-ReasonRAG",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
