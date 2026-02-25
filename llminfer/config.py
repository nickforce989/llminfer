"""
Configuration dataclasses for llminfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class QuantMode(str, Enum):
    NONE = "none"
    INT8 = "int8"       # bitsandbytes LLM.int8()
    NF4  = "nf4"        # 4-bit NormalFloat (QLoRA style)
    FP4  = "fp4"        # 4-bit FloatingPoint


class Backend(str, Enum):
    EAGER    = "eager"     # Standard PyTorch
    COMPILED = "compiled"  # torch.compile (Inductor)
    VLLM     = "vllm"      # vLLM PagedAttention


class CacheEviction(str, Enum):
    NONE   = "none"    # Keep everything (OOM risk)
    FIFO   = "fifo"    # Evict oldest sequence
    LRU    = "lru"     # Least-recently-used


@dataclass
class QuantConfig:
    """Controls weight quantization."""
    mode: QuantMode = QuantMode.NONE
    # For 4-bit: whether to use double quantization (saves ~0.4 bits/param)
    double_quant: bool = True
    # Compute dtype while using quantized weights
    compute_dtype: str = "bfloat16"   # "float16" or "bfloat16"

    def to_bnb_kwargs(self) -> dict:
        """Return kwargs for BitsAndBytesConfig."""
        if self.mode == QuantMode.NONE:
            return {}
        import torch
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map.get(self.compute_dtype, torch.bfloat16)
        if self.mode == QuantMode.INT8:
            return {"load_in_8bit": True}
        # 4-bit modes
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": self.mode.value,         # "nf4" or "fp4"
            "bnb_4bit_use_double_quant": self.double_quant,
            "bnb_4bit_compute_dtype": compute_dtype,
        }


@dataclass
class CacheConfig:
    """Controls KV-cache behaviour."""
    max_seqs: int = 64            # Max concurrent sequences in cache
    max_seq_len: int = 2048       # Max tokens per sequence slot
    eviction: CacheEviction = CacheEviction.LRU
    # Prefix caching: reuse KV blocks for common prompt prefixes
    enable_prefix_cache: bool = True
    # GPU memory fraction to allocate for the KV cache
    gpu_memory_fraction: float = 0.15


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    model_name: str = "facebook/opt-125m"
    backend: Backend = Backend.EAGER
    quant: QuantConfig = field(default_factory=QuantConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

    # Batching
    max_batch_size: int = 8
    # How long (ms) to wait before flushing an incomplete batch
    batch_timeout_ms: float = 20.0

    # Device
    device: str = "cuda"   # "cuda", "cuda:0", "cpu"
    dtype: str = "auto"    # "auto", "float16", "bfloat16", "float32"
    # Optional speculative-decoding assistant model (HF generate assistant_model)
    assistant_model_name: Optional[str] = None

    # torch.compile settings (only used if backend == COMPILED)
    compile_mode: str = "reduce-overhead"   # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = True
    # If compiled execution fails at runtime, automatically fall back to eager.
    compile_fallback_to_eager: bool = True
