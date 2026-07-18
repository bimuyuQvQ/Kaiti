"""Construct legacy ETC with research-only deployment controls."""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


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
