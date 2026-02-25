"""
Eager PyTorch backend.

Uses standard HuggingFace transformers with:
  - Optional BitsAndBytes quantization (int8, nf4, fp4)
  - KV cache reuse across decode steps
  - Prefix cache for shared prompt prefixes
  - Token streaming via TextIteratorStreamer
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)

from llminfer.backends.base import BaseBackend
from llminfer.config import EngineConfig, QuantMode
from llminfer.kv_cache import KVCacheManager
from llminfer.request import GenerationRequest, GenerationResult, StreamChunk, TokenStats

logger = logging.getLogger(__name__)


class EagerBackend(BaseBackend):
    """Standard PyTorch eager-mode backend."""

    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__(cfg)
        self._kv_cache = KVCacheManager(cfg.cache)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading model %s  quant=%s", self.cfg.model_name, self.cfg.quant.mode)

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config
        bnb_kwargs = self.cfg.quant.to_bnb_kwargs()
        bnb_config = BitsAndBytesConfig(**bnb_kwargs) if bnb_kwargs else None

        # dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.cfg.dtype, "auto")

        load_kwargs: dict = {
            "device_map": self.cfg.device if not bnb_config else "auto",
            "torch_dtype": torch_dtype,
        }
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, **load_kwargs
        )
        self._model.eval()
        logger.info("Model loaded  params=%.1fM", self._count_params() / 1e6)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_gen_config(self, req: GenerationRequest) -> dict:
        return dict(
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0 and req.temperature != 1.0,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            use_cache=True,
        )

    def _tokenize_batch(self, prompts: List[str]) -> dict:
        return self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.cache.max_seq_len,
        ).to(self.cfg.device)

    # ------------------------------------------------------------------
    # Batch generate
    # ------------------------------------------------------------------

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self.is_loaded:
            self.load()

        prompts = [r.prompt for r in requests]
        inputs = self._tokenize_batch(prompts)
        prompt_len = inputs["input_ids"].shape[1]

        # Check prefix cache for single-request case
        cache_hit = False
        past_kv = None
        if len(requests) == 1 and requests[0].prefix_key:
            past_kv = self._kv_cache.lookup_prefix(requests[0].prefix_key)
            cache_hit = past_kv is not None

        t_start = time.monotonic()
        gen_kwargs = self._build_gen_config(requests[0])  # use first req params for batch
        if past_kv is not None:
            gen_kwargs["past_key_values"] = past_kv

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000

        results = []
        for i, req in enumerate(requests):
            # Decode only newly generated tokens
            gen_ids = outputs[i][prompt_len:]
            gen_text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_tokens = len(gen_ids)

            stats = TokenStats(
                prompt_tokens=prompt_len,
                generated_tokens=gen_tokens,
                total_latency_ms=total_ms,
                throughput_tokens_per_sec=gen_tokens / max(total_ms / 1000, 1e-9),
                cache_hit=cache_hit,
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

    # ------------------------------------------------------------------
    # Streaming (single request)
    # ------------------------------------------------------------------

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        if not self.is_loaded:
            self.load()

        inputs = self._tokenize_batch([request.prompt])

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **self._build_gen_config(request),
            "streamer": streamer,
        }

        # Run generate in background thread so we can yield from main thread
        t_start = time.monotonic()
        first_token_time: list[float] = []

        def _generate():
            with torch.inference_mode():
                self._model.generate(**inputs, **gen_kwargs)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        token_count = 0
        for text_piece in streamer:
            now = time.monotonic()
            if not first_token_time:
                first_token_time.append(now)
            token_count += 1
            yield StreamChunk(
                request_id=request.request_id,
                token=text_piece,
                token_id=-1,
                is_final=False,
            )

        thread.join()
        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = ((first_token_time[0] if first_token_time else t_end) - t_start) * 1000

        stats = TokenStats(
            generated_tokens=token_count,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            throughput_tokens_per_sec=token_count / max(total_ms / 1000, 1e-9),
        )
        yield StreamChunk(
            request_id=request.request_id,
            token="",
            token_id=-1,
            is_final=True,
            stats=stats,
        )
