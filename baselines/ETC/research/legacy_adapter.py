"""Construct legacy ETC with research-only deployment controls."""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def install_last_layer_attention_capture(model: Any) -> None:
    """Return only last-layer attention while preserving legacy eager math.

    Transformers normally accumulates every layer's attention whenever the top
    level requests attentions. ETC consumes only the final element. During that
    diagnostic call we run eager attention exactly as legacy does, but discard
    earlier layer matrices immediately and attach only the captured final one.
    """

    if getattr(model, "_cura_last_attention_installed", False):
        return
    last_layer = model.model.layers[-1]
    last_attention = last_layer.self_attn
    model_forward_name = "_old_forward" if hasattr(model, "_old_forward") else "forward"
    attention_forward_name = "_old_forward" if hasattr(last_attention, "_old_forward") else "forward"
    original_model_forward = getattr(model, model_forward_name)
    original_attention_forward = getattr(last_attention, attention_forward_name)

    def memory_efficient_forward(*args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("output_attentions", False):
            return original_model_forward(*args, **kwargs)

        captured: Dict[str, Any] = {}
        original_implementation = model.config._attn_implementation
        attention_config = last_attention.config
        original_attention_implementation = attention_config._attn_implementation

        def capture_last_attention(*attention_args: Any, **attention_kwargs: Any) -> Any:
            outputs = original_attention_forward(*attention_args, **attention_kwargs)
            captured["attention"] = outputs[1]
            return outputs

        try:
            model.config._attn_implementation = "eager"
            attention_config._attn_implementation = "eager"
            setattr(last_attention, attention_forward_name, capture_last_attention)
            forwarded = dict(kwargs)
            forwarded["output_attentions"] = False
            outputs = original_model_forward(*args, **forwarded)
        finally:
            setattr(last_attention, attention_forward_name, original_attention_forward)
            model.config._attn_implementation = original_implementation
            attention_config._attn_implementation = original_attention_implementation

        if "attention" not in captured:
            raise RuntimeError("未能捕获 Llama 最后一层注意力")
        attention_tuple = (captured["attention"],)
        if hasattr(outputs, "__setitem__"):
            # Accelerate's post-forward hook moves ModelOutput by dictionary
            # keys. A plain attribute assignment would be lost because the
            # original None-valued `attentions` field is absent from the keys.
            outputs["attentions"] = attention_tuple
        else:
            outputs.attentions = attention_tuple
        return outputs

    setattr(model, model_forward_name, memory_efficient_forward)
    model._cura_last_attention_installed = True


def resolve_max_memory(max_memory_gib: int, device_count: Optional[int] = None) -> Dict[int, str]:
    if max_memory_gib <= 0:
        raise ValueError("model_max_memory_gib 必须为正数")
    if device_count is None:
        import torch

        device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("ETC 研究运行需要至少一张可见 CUDA GPU")
    return {index: f"{max_memory_gib}GiB" for index in range(device_count)}


def build_research_etc(args: Any, research_config: Dict[str, Any]) -> Any:
    """Reuse ETC algorithms while leaving headroom for attention diagnostics.

    The released `Generator.__init__` hard-codes `device_map="auto"` without a
    memory ceiling.  CURA observes more online states, so it needs explicit GPU
    headroom.  We construct the same legacy objects without editing their files.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generate import Counter, ETC, Generator

    max_memory_gib = int(research_config.get("model_max_memory_gib", 18))
    max_memory = resolve_max_memory(max_memory_gib)
    logger.info("研究模型分片上限: %s", max_memory)

    generator = Generator.__new__(Generator)
    generator.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator.model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        max_memory=max_memory,
    )
    install_last_layer_attention_capture(generator.model)
    generator.space_token = "Ġ" if "llama-3" in args.model_name_or_path.lower() else "▁"
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    generator.tokens_cannot_merged = {
        generator.tokenizer.convert_ids_to_tokens(generator.tokenizer.encode("0" + character)[-1:])[0]
        for character in string.whitespace + string.punctuation
    } | {
        generator.space_token,
        generator.tokenizer.bos_token,
        generator.tokenizer.eos_token,
    }

    etc = ETC.__new__(ETC)
    for key, value in args.__dict__.items():
        setattr(etc, key, value)
    etc.generator = generator
    etc.tokenizer = generator.tokenizer
    etc.model = generator.model
    etc.counter = Counter()
    return etc
