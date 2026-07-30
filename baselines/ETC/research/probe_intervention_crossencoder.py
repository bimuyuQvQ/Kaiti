"""Small BGE cross-encoder probe for KEEP versus FULL_RESTART learnability."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .probe_intervention_learnability import (
    _aggregate,
    _choose_policy,
    _group_splits,
    load_rows,
)


CROSS_ENCODER_VERSION = "bge_keep_restart_multitask_probe_v1"


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _text_key(variant: str) -> str:
    if variant == "full":
        return "text_full"
    if variant == "no_prefix":
        return "text_no_prefix"
    raise ValueError(f"未知 variant：{variant}")


def _tokenize_rows(rows, tokenizer, variant: str, max_length: int):
    import torch

    encoded = tokenizer(
        [str(row[_text_key(variant)]) for row in rows],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded["delta_labels"] = torch.tensor(
        [float(row["labels"]["restart_gain_over_keep"]) for row in rows],
        dtype=torch.float32,
    )
    encoded["harm_labels"] = torch.tensor(
        [float(row["labels"]["restart_harm"]) for row in rows], dtype=torch.float32
    )
    return encoded


class _BatchDataset:
    def __init__(self, tensors: Mapping[str, Any], indices: Sequence[int]):
        self.tensors = tensors
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Dict[str, Any]:
        index = self.indices[position]
        return {key: value[index] for key, value in self.tensors.items()}


def _build_model(model_path: str, unfreeze_last_layers: int):
    import torch
    from transformers import AutoModel

    class MultiTaskProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = torch.nn.Dropout(0.1)
            self.value_head = torch.nn.Linear(hidden, 1)
            self.harm_head = torch.nn.Linear(hidden, 1)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if layers is None:
                raise ValueError("基础模型没有可识别的 encoder.layer")
            for layer in layers[-unfreeze_last_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            output = self.encoder(**kwargs)
            representation = self.dropout(output.last_hidden_state[:, 0])
            return (
                self.value_head(representation).squeeze(-1),
                self.harm_head(representation).squeeze(-1),
            )

    return MultiTaskProbe()


def _loader(tensors, indices, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        _BatchDataset(tensors, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _predict(model, loader, device) -> Tuple[List[float], List[float], float]:
    import torch

    model.eval()
    deltas: List[float] = []
    harms: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            delta, harm_logit = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            losses.extend(
                torch.square(delta - batch["delta_labels"]).detach().cpu().tolist()
            )
            deltas.extend(delta.detach().cpu().tolist())
            harms.extend(torch.sigmoid(harm_logit).detach().cpu().tolist())
    return deltas, harms, sum(losses) / len(losses)


def _train_model(
    model_path: str,
    tensors,
    train_indices: Sequence[int],
    validation_indices: Sequence[int] | None,
    device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    head_learning_rate: float,
    harm_weight: float,
    unfreeze_last_layers: int,
    seed: int,
) -> Tuple[Any, int, Dict[str, float]]:
    import torch

    _seed_everything(seed)
    model = _build_model(model_path, unfreeze_last_layers).to(device)
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ],
        weight_decay=0.01,
    )
    train_loader = _loader(tensors, train_indices, batch_size, shuffle=True)
    positives = sum(float(tensors["harm_labels"][index]) for index in train_indices)
    negatives = len(train_indices) - positives
    positive_weight = torch.tensor(
        [negatives / max(1.0, positives)], dtype=torch.float32, device=device
    )
    harm_loss = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    value_loss = torch.nn.SmoothL1Loss()
    best_epoch = epochs
    best_validation = float("inf")
    best_state = None
    stale = 0
    history = {}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        examples = 0
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            delta, harm_logit = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            loss = value_loss(delta, batch["delta_labels"]) + harm_weight * harm_loss(
                harm_logit, batch["harm_labels"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(delta)
            examples += len(delta)
        history[f"train_loss_epoch_{epoch}"] = total / examples
        if validation_indices is None:
            continue
        _, _, validation_mse = _predict(
            model,
            _loader(tensors, validation_indices, batch_size * 2, shuffle=False),
            device,
        )
        history[f"validation_mse_epoch_{epoch}"] = validation_mse
        if validation_mse < best_validation - 1e-6:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = float(best_epoch)
    history["best_validation_mse"] = best_validation
    return model, best_epoch, history


def _inner_split(rows, outer_train_indices: Sequence[int], folds: int):
    outer_rows = [rows[int(index)] for index in outer_train_indices]
    train_relative, validation_relative = _group_splits(outer_rows, folds)[0]
    train_indices = [int(outer_train_indices[int(index)]) for index in train_relative]
    validation_indices = [
        int(outer_train_indices[int(index)]) for index in validation_relative
    ]
    return train_indices, validation_indices


def _refit_full_outer_train(
    model_path,
    tensors,
    train_indices,
    device,
    epochs,
    batch_size,
    learning_rate,
    head_learning_rate,
    harm_weight,
    unfreeze_last_layers,
    seed,
):
    model, _, history = _train_model(
        model_path=model_path,
        tensors=tensors,
        train_indices=train_indices,
        validation_indices=None,
        device=device,
        epochs=epochs,
        patience=epochs + 1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_learning_rate=head_learning_rate,
        harm_weight=harm_weight,
        unfreeze_last_layers=unfreeze_last_layers,
        seed=seed,
    )
    return model, history


def run_cross_encoder_probe(
    rows: Sequence[Dict[str, Any]],
    model_path: str,
    variants: Sequence[str] = ("no_prefix", "full"),
    outer_folds: int = 5,
    inner_folds: int = 5,
    epochs: int = 8,
    patience: int = 2,
    batch_size: int = 8,
    max_length: int = 512,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-3,
    harm_weight: float = 0.2,
    unfreeze_last_layers: int = 2,
    bootstrap_samples: int = 10_000,
    seed: int = 20260730,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("cross-encoder probe 需要 CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    outer_splits = _group_splits(rows, outer_folds)
    records: List[Dict[str, Any]] = []
    fold_history: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    for variant in variants:
        tensors = _tokenize_rows(rows, tokenizer, variant, max_length)
        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            inner_train, inner_validation = _inner_split(rows, outer_train, inner_folds)
            selection_model, best_epoch, selection_history = _train_model(
                model_path=model_path,
                tensors=tensors,
                train_indices=inner_train,
                validation_indices=inner_validation,
                device=device,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                head_learning_rate=head_learning_rate,
                harm_weight=harm_weight,
                unfreeze_last_layers=unfreeze_last_layers,
                seed=seed + fold,
            )
            dev_delta, dev_harm, _ = _predict(
                selection_model,
                _loader(tensors, inner_validation, batch_size * 2, shuffle=False),
                device,
            )
            dev_rows = [rows[index] for index in inner_validation]
            value_policy = _choose_policy(dev_rows, dev_delta, dev_harm, risk_aware=False)
            risk_policy = _choose_policy(dev_rows, dev_delta, dev_harm, risk_aware=True)
            del selection_model
            torch.cuda.empty_cache()
            final_model, refit_history = _refit_full_outer_train(
                model_path,
                tensors,
                [int(index) for index in outer_train],
                device,
                best_epoch,
                batch_size,
                learning_rate,
                head_learning_rate,
                harm_weight,
                unfreeze_last_layers,
                seed + 100 + fold,
            )
            test_delta, test_harm, test_mse = _predict(
                final_model,
                _loader(tensors, outer_test, batch_size * 2, shuffle=False),
                device,
            )
            del final_model
            torch.cuda.empty_cache()
            outer_train_rows = [rows[int(index)] for index in outer_train]
            keep_train = sum(row["labels"]["keep_f1"] for row in outer_train_rows) / len(
                outer_train_rows
            )
            restart_train = sum(
                row["labels"]["restart_f1"] for row in outer_train_rows
            ) / len(outer_train_rows)
            fixed_action = "restart" if restart_train >= keep_train else "keep"
            fold_history[variant].append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "test_value_mse": test_mse,
                    "selection_history": selection_history,
                    "refit_history": refit_history,
                    "value_policy": value_policy,
                    "risk_policy": risk_policy,
                }
            )
            for index, delta_prediction, harm_probability in zip(
                outer_test, test_delta, test_harm
            ):
                row = rows[int(index)]
                record: Dict[str, Any] = {
                    "probe_version": CROSS_ENCODER_VERSION,
                    "variant": variant,
                    "outer_fold": fold,
                    "sample_index": int(row["sample_index"]),
                    "qid": str(row["qid"]),
                    "state_id": str(row["state_id"]),
                    "query_source": str(row["query_source"]),
                    "delta_prediction": float(delta_prediction),
                    "harm_probability": float(harm_probability),
                    "keep_f1": float(row["labels"]["keep_f1"]),
                    "restart_f1": float(row["labels"]["restart_f1"]),
                    "restart_harm": int(row["labels"]["restart_harm"]),
                    "restart_rescue": int(row["labels"]["restart_rescue"]),
                    "fold_best_fixed_action": fixed_action,
                    "fixed_score": float(row["labels"][f"{fixed_action}_f1"]),
                    "oracle_score": max(
                        float(row["labels"]["keep_f1"]),
                        float(row["labels"]["restart_f1"]),
                    ),
                }
                for policy_name, policy in (
                    ("value", value_policy),
                    ("risk", risk_policy),
                ):
                    adjusted = float(delta_prediction) - float(
                        policy["lambda_harm"]
                    ) * float(harm_probability)
                    action = (
                        "restart" if adjusted > float(policy["threshold"]) else "keep"
                    )
                    record[f"{policy_name}_selected_action"] = action
                    record[f"{policy_name}_selected_score"] = float(
                        row["labels"][f"{action}_f1"]
                    )
                    record[f"{policy_name}_lambda_harm"] = policy["lambda_harm"]
                    record[f"{policy_name}_threshold"] = policy["threshold"]
                records.append(record)

    variant_reports = {}
    for variant in variants:
        variant_records = [record for record in records if record["variant"] == variant]
        variant_reports[variant] = {
            "value_policy": _aggregate(
                variant_records, "value", bootstrap_samples=bootstrap_samples
            ),
            "risk_policy": _aggregate(
                variant_records, "risk", bootstrap_samples=bootstrap_samples
            ),
            "fold_history": fold_history[variant],
        }
    full_recovery = variant_reports.get("full", {}).get("risk_policy", {}).get(
        "normalized_oracle_recovery"
    )
    no_prefix_recovery = variant_reports.get("no_prefix", {}).get(
        "risk_policy", {}
    ).get("normalized_oracle_recovery")
    report = {
        "probe_version": CROSS_ENCODER_VERSION,
        "rows": len(rows),
        "qids": len({row["qid"] for row in rows}),
        "model_path": model_path,
        "variants": variant_reports,
        "full_minus_no_prefix_recovery": (
            float(full_recovery) - float(no_prefix_recovery)
            if full_recovery is not None and no_prefix_recovery is not None
            else None
        ),
        "configuration": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "head_learning_rate": head_learning_rate,
            "harm_weight": harm_weight,
            "unfreeze_last_layers": unfreeze_last_layers,
            "seed": seed,
        },
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "interpretation": (
            "每个外折只在该折训练 qid 上选择 epoch、阈值和 harm 权重；"
            "测试 qid 始终隔离。BGE 只解冻最后若干层，属于小型语义可学习性 probe。"
        ),
    }
    return report, records


def _write_json_exclusive(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--variant", action="append", choices=["no_prefix", "full"])
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--harm_weight", type=float, default=0.2)
    parser.add_argument("--unfreeze_last_layers", type=int, default=2)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof_output", required=True)
    args = parser.parse_args()
    variants = args.variant or ["no_prefix", "full"]
    report, records = run_cross_encoder_probe(
        load_rows(args.supervision),
        model_path=args.model_path,
        variants=variants,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        harm_weight=args.harm_weight,
        unfreeze_last_layers=args.unfreeze_last_layers,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_json_exclusive(args.output, report)
    _write_jsonl_exclusive(args.oof_output, records)
    compact = {
        "probe_version": report["probe_version"],
        "rows": report["rows"],
        "qids": report["qids"],
        "full_minus_no_prefix_recovery": report["full_minus_no_prefix_recovery"],
        "variants": {
            name: {
                policy: {
                    key: result[policy][key]
                    for key in (
                        "selected_mean_f1",
                        "fixed_mean_f1",
                        "oracle_mean_f1",
                        "normalized_oracle_recovery",
                        "selected_harm_rows",
                        "selected_rescue_rows",
                        "harm_auprc",
                    )
                }
                for policy in ("value_policy", "risk_policy")
            }
            for name, result in report["variants"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
