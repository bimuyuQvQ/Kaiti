"""Compare legacy all-layer attention with last-layer-only capture on a short input."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_max_memory_gib", type=int, default=14)
    parser.add_argument("--text", default="Question: Who wrote Hamlet? Answer:")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .legacy_adapter import install_last_layer_attention_capture, resolve_max_memory

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        max_memory=resolve_max_memory(args.model_max_memory_gib),
    )
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    legacy = model(
        input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
        num_logits_to_keep=1,
    )
    legacy_attention = legacy.attentions[-1].detach().float().cpu()
    legacy_logits = legacy.logits.detach().float().cpu()
    del legacy
    torch.cuda.empty_cache()

    install_last_layer_attention_capture(model)
    adapted = model(
        input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
        num_logits_to_keep=1,
    )
    adapted_attention = adapted.attentions[-1].detach().float().cpu()
    adapted_logits = adapted.logits.detach().float().cpu()
    result = {
        "attention_shape_equal": legacy_attention.shape == adapted_attention.shape,
        "attention_max_abs_diff": float((legacy_attention - adapted_attention).abs().max().item()),
        "attention_allclose_1e_6": bool(torch.allclose(legacy_attention, adapted_attention, atol=1e-6, rtol=1e-6)),
        "logits_max_abs_diff": float((legacy_logits - adapted_logits).abs().max().item()),
        "logits_allclose_1e_6": bool(torch.allclose(legacy_logits, adapted_logits, atol=1e-6, rtol=1e-6)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["attention_allclose_1e_6"] or not result["logits_allclose_1e_6"]:
        raise SystemExit(2)


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        main()
