"""
vLLM backend wrapper.

vLLM uses PagedAttention for near-zero KV-cache fragmentation and
continuous batching, making it the current state-of-the-art for
high-throughput LLM serving.

This backend wraps vllm.LLM to match the llminfer BaseBackend interface,
enabling fair apples-to-apples benchmarking against our eager/compiled
implementations.

Requirements
------------
pip install vllm   (separate install, not in base requirements)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, List

from llminfer.backends.base import BaseBackend
from llminfer.config import EngineConfig, QuantMode
from llminfer.request import GenerationRequest, GenerationResult, StreamChunk, TokenStats

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM PagedAttention backend."""

    QUANT_MAP = {
        QuantMode.INT8: "bitsandbytes",
        QuantMode.NF4: "bitsandbytes",
        QuantMode.FP4: "bitsandbytes",
        QuantMode.NONE: None,
    }

    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__(cfg)
        self._llm = None

    def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with:\n"
                "  pip install vllm\n"
                "Note: vLLM requires CUDA and Linux."
            )

        logger.info("Loading vLLM engine  model=%s", self.cfg.model_name)

        quant = self.QUANT_MAP.get(self.cfg.quant.mode)
        kwargs: dict = {
            "model": self.cfg.model_name,
            "dtype": self.cfg.dtype if self.cfg.dtype != "auto" else "auto",
            "gpu_memory_utilization": self.cfg.cache.gpu_memory_fraction + 0.75,  # vLLM uses most of VRAM
            "max_model_len": self.cfg.cache.max_seq_len,
            "trust_remote_code": True,
        }
        if quant:
            kwargs["quantization"] = quant

        self._llm = LLM(**kwargs)
        logger.info("vLLM engine ready")

    def _make_sampling_params(self, req: GenerationRequest):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self.is_loaded:
            self.load()

        prompts = [r.prompt for r in requests]
        sampling_params = [self._make_sampling_params(r) for r in requests]

        t_start = time.monotonic()
        outputs = self._llm.generate(prompts, sampling_params)
        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000

        results = []
        for req, out in zip(requests, outputs):
            gen_text = out.outputs[0].text
            gen_tokens = len(out.outputs[0].token_ids)
            prompt_tokens = len(out.prompt_token_ids)

            stats = TokenStats(
                prompt_tokens=prompt_tokens,
                generated_tokens=gen_tokens,
                total_latency_ms=total_ms,
                throughput_tokens_per_sec=gen_tokens / max(total_ms / 1000, 1e-9),
            )
            results.append(
                GenerationResult(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    generated_text=gen_text,
                    stats=stats,
                )
            )

        return results

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """
        vLLM streaming via async generator.
        We run the async engine synchronously here for interface compatibility.
        """
        if not self.is_loaded:
            self.load()

        import asyncio

        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        # Note: for production streaming, use vllm.AsyncLLMEngine directly.
        # Here we fake streaming by chunking the completed output.
        results = self.generate([request])
        result = results[0]
        words = result.generated_text.split(" ")

        t_start = time.monotonic()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield StreamChunk(
                request_id=request.request_id,
                token=token,
                token_id=-1,
                is_final=False,
            )

        yield StreamChunk(
            request_id=request.request_id,
            token="",
            token_id=-1,
            is_final=True,
            stats=result.stats,
        )

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None
