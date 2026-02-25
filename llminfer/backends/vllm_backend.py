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

import asyncio
import inspect
import logging
import queue
import threading
import time
from typing import Iterator, List, Optional

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
        self._async_engine = None
        self._engine_args = None

    def load(self) -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with:\n"
                "  pip install vllm\n"
                "Note: vLLM requires CUDA and Linux."
            )

        logger.info("Loading vLLM engine  model=%s", self.cfg.model_name)

        quant = self.QUANT_MAP.get(self.cfg.quant.mode)
        gpu_util = min(0.98, max(self.cfg.cache.gpu_memory_fraction + 0.75, 0.1))
        kwargs: dict = {
            "model": self.cfg.model_name,
            "dtype": self.cfg.dtype if self.cfg.dtype != "auto" else "auto",
            "gpu_memory_utilization": gpu_util,  # vLLM typically uses most of VRAM
            "max_model_len": self.cfg.cache.max_seq_len,
            "trust_remote_code": self.cfg.hf_trust_remote_code,
            "tensor_parallel_size": max(1, int(self.cfg.tensor_parallel_size)),
            "pipeline_parallel_size": max(1, int(self.cfg.pipeline_parallel_size)),
        }
        if quant:
            kwargs["quantization"] = quant
        if self.cfg.assistant_model_name:
            kwargs["speculative_model"] = self.cfg.assistant_model_name
        if self.cfg.speculative_num_assistant_tokens is not None:
            kwargs["num_speculative_tokens"] = int(self.cfg.speculative_num_assistant_tokens)
        if self.cfg.hf_revision:
            kwargs["revision"] = self.cfg.hf_revision
            kwargs["tokenizer_revision"] = self.cfg.hf_revision
        if self.cfg.hf_token:
            # Different vLLM versions expose either `token` or `hf_token`.
            kwargs["token"] = self.cfg.hf_token
            kwargs["hf_token"] = self.cfg.hf_token
        if self.cfg.hf_cache_dir:
            kwargs["download_dir"] = self.cfg.hf_cache_dir

        # Keep compatibility across vLLM versions by filtering unsupported args.
        sig = inspect.signature(AsyncEngineArgs.__init__)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        self._engine_args = AsyncEngineArgs(**kwargs)
        self._async_engine = AsyncLLMEngine.from_engine_args(self._engine_args)
        logger.info("vLLM engine ready")

    def _make_sampling_params(self, req: GenerationRequest):
        from vllm import SamplingParams
        kwargs = dict(
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop_sequences,
        )
        if req.bad_words:
            kwargs["bad_words"] = req.bad_words
        if req.seed is not None:
            kwargs["seed"] = req.seed
        return SamplingParams(**kwargs)

    @staticmethod
    def _run_async(coro):
        holder = {"result": None, "error": None}

        def _runner():
            try:
                holder["result"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - passthrough
                holder["error"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if holder["error"] is not None:
            raise holder["error"]
        return holder["result"]

    @staticmethod
    def _apply_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
        if not stop_sequences:
            return text
        cut = len(text)
        for stop in stop_sequences:
            idx = text.find(stop)
            if idx != -1 and idx < cut:
                cut = idx
        return text[:cut]

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self.is_loaded:
            self.load()

        async def _generate_one(req: GenerationRequest):
            sampling_params = self._make_sampling_params(req)
            t_start = time.monotonic()
            final_output = None
            async for out in self._async_engine.generate(req.prompt, sampling_params, req.request_id):
                final_output = out
            total_ms = (time.monotonic() - t_start) * 1000
            return req, final_output, total_ms

        async def _run_all():
            tasks = [asyncio.create_task(_generate_one(r)) for r in requests]
            return await asyncio.gather(*tasks)

        outputs = self._run_async(_run_all())

        results = []
        for req, out, total_ms in outputs:
            gen_text = out.outputs[0].text if out else ""
            gen_text = self._apply_stop_sequences(gen_text, req.stop_sequences)
            gen_tokens = len(out.outputs[0].token_ids) if out else 0
            prompt_tokens = len(out.prompt_token_ids) if out else 0

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
        if not self.is_loaded:
            self.load()

        out_q: queue.Queue = queue.Queue()
        sampling_params = self._make_sampling_params(request)

        def _run_stream() -> None:
            async def _produce() -> None:
                t_start = time.monotonic()
                first_token_time = None
                emitted_text = ""
                final_output = None

                async for out in self._async_engine.generate(request.prompt, sampling_params, request.request_id):
                    final_output = out
                    latest_text = out.outputs[0].text
                    clipped = self._apply_stop_sequences(latest_text, request.stop_sequences)
                    new_text = clipped[len(emitted_text) :]
                    if new_text:
                        emitted_text = clipped
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        out_q.put(("token", new_text))

                    if len(clipped) < len(latest_text):
                        # Stop sequence matched. Abort remaining generation.
                        try:
                            await self._async_engine.abort(request.request_id)
                        except Exception:
                            pass
                        break

                t_end = time.monotonic()
                total_ms = (t_end - t_start) * 1000
                gen_tokens = len(final_output.outputs[0].token_ids) if final_output else 0
                prompt_tokens = len(final_output.prompt_token_ids) if final_output else 0
                ttft_ms = ((first_token_time or t_end) - t_start) * 1000

                stats = TokenStats(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=gen_tokens,
                    time_to_first_token_ms=ttft_ms,
                    total_latency_ms=total_ms,
                    throughput_tokens_per_sec=gen_tokens / max(total_ms / 1000, 1e-9),
                )
                out_q.put(("final", stats))

            try:
                asyncio.run(_produce())
            except Exception as exc:  # pragma: no cover - passthrough
                out_q.put(("error", exc))

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        while True:
            kind, payload = out_q.get()
            if kind == "token":
                token = payload
                yield StreamChunk(
                    request_id=request.request_id,
                    token=token,
                    token_id=-1,
                    is_final=False,
                )
                continue
            if kind == "final":
                stats = payload
                yield StreamChunk(
                    request_id=request.request_id,
                    token="",
                    token_id=-1,
                    is_final=True,
                    stats=stats,
                )
                break
            if kind == "error":
                raise payload

        thread.join(timeout=0.1)

    @property
    def is_loaded(self) -> bool:
        return self._async_engine is not None

    def unload(self) -> None:
        if self._async_engine is not None:
            shutdown = getattr(self._async_engine, "shutdown_background_loop", None)
            if callable(shutdown):
                shutdown()
        self._async_engine = None
        self._engine_args = None
        super().unload()
