"""
InferenceEngine — the main public interface for llminfer.

Selects the right backend, manages the KV cache, handles
batching, and exposes both sync and async generation APIs.

Quick start
-----------
from llminfer import InferenceEngine, EngineConfig
from llminfer.config import Backend, QuantConfig, QuantMode

cfg = EngineConfig(
    model_name="facebook/opt-1.3b",
    backend=Backend.EAGER,
    quant=QuantConfig(mode=QuantMode.NF4),
)
engine = InferenceEngine(cfg)
engine.load()

result = engine.run("Tell me about transformers")
print(result.generated_text)

for chunk in engine.stream("Tell me about transformers"):
    print(chunk.token, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Iterator, List, Optional

from llminfer.backends.base import BaseBackend
from llminfer.backends.compiled import CompiledBackend
from llminfer.backends.eager import EagerBackend
from llminfer.config import Backend, EngineConfig
from llminfer.kv_cache import KVCacheManager
from llminfer.request import GenerationRequest, GenerationResult, StreamChunk

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-level inference engine.

    Responsibilities
    ----------------
    - Select and load the appropriate backend
    - Provide simple .run() / .run_batch() / .stream() APIs
    - Expose cache and backend stats
    """

    def __init__(self, cfg: Optional[EngineConfig] = None) -> None:
        self.cfg = cfg or EngineConfig()
        self._backend: Optional[BaseBackend] = None
        self._kv_cache = KVCacheManager(self.cfg.cache)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "InferenceEngine":
        """Load the model. Returns self for chaining."""
        if self._backend is not None and self._backend.is_loaded:
            return self
        self._backend = self._build_backend()
        self._backend.load()
        logger.info(
            "InferenceEngine ready  backend=%s  model=%s",
            self.cfg.backend.value,
            self.cfg.model_name,
        )
        return self

    def _build_backend(self) -> BaseBackend:
        if self.cfg.backend == Backend.EAGER:
            return EagerBackend(self.cfg)
        elif self.cfg.backend == Backend.COMPILED:
            return CompiledBackend(self.cfg)
        elif self.cfg.backend == Backend.VLLM:
            from llminfer.backends.vllm_backend import VLLMBackend
            return VLLMBackend(self.cfg)
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

    def unload(self) -> None:
        if self._backend:
            self._backend.unload()
            self._backend = None

    def warmup(self, prompt: str = "Hello", n: int = 2) -> None:
        """Trigger JIT compilation / CUDA graph capture (compiled backend only)."""
        if isinstance(self._backend, CompiledBackend):
            self._backend.warmup(prompt=prompt, n=n)

    # ------------------------------------------------------------------
    # Generation APIs
    # ------------------------------------------------------------------

    def run(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate a completion for a single prompt.

        Parameters
        ----------
        prompt : str
        **kwargs : overrides for GenerationRequest fields
        """
        self._ensure_loaded()
        req = GenerationRequest(prompt=prompt, **self._with_generation_defaults(kwargs))
        results = self._backend.generate([req])
        return results[0]

    def run_batch(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """
        Generate completions for a list of prompts.
        Automatically splits into batches of max_batch_size.
        """
        self._ensure_loaded()
        req_kwargs = self._with_generation_defaults(kwargs)
        requests = [GenerationRequest(prompt=p, **req_kwargs) for p in prompts]
        results: List[GenerationResult] = []

        # Process in chunks of max_batch_size
        bs = self.cfg.max_batch_size
        for i in range(0, len(requests), bs):
            chunk = requests[i : i + bs]
            results.extend(self._backend.generate(chunk))

        return results

    def run_requests(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Run a heterogeneous request batch (each request can have distinct params).
        """
        self._ensure_loaded()
        return self._backend.generate(requests)

    async def run_requests_continuous_async(
        self,
        requests: List[GenerationRequest],
        max_queue_size: Optional[int] = None,
    ) -> List[GenerationResult]:
        """
        Run requests through the continuous-batching scheduler.

        Unlike `run_requests`, requests can keep arriving while prior requests
        are being processed, emulating serving-style traffic.
        """
        if not requests:
            return []
        self._ensure_loaded()
        from llminfer.serving import ContinuousBatchScheduler

        queue_size = max_queue_size or max(1024, len(requests) * 2)
        scheduler = ContinuousBatchScheduler(
            engine=self,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            max_queue_size=queue_size,
        )
        await scheduler.start()
        try:
            tasks = [asyncio.create_task(scheduler.submit(req)) for req in requests]
            return await asyncio.gather(*tasks)
        finally:
            await scheduler.stop()

    def run_requests_continuous(
        self,
        requests: List[GenerationRequest],
        max_queue_size: Optional[int] = None,
    ) -> List[GenerationResult]:
        """
        Synchronous wrapper around `run_requests_continuous_async`.
        """
        return self._run_async(
            self.run_requests_continuous_async(
                requests=requests,
                max_queue_size=max_queue_size,
            )
        )

    def run_batch_continuous(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """
        Convenience continuous-batching API for homogeneous prompt lists.
        """
        req_kwargs = self._with_generation_defaults(kwargs)
        requests = [GenerationRequest(prompt=p, **req_kwargs) for p in prompts]
        return self.run_requests_continuous(requests=requests)

    def stream(self, prompt: str, **kwargs) -> Iterator[StreamChunk]:
        """
        Stream token-by-token for a single prompt.

        Usage
        -----
        for chunk in engine.stream("Tell me a story"):
            if not chunk.is_final:
                print(chunk.token, end="", flush=True)
        """
        self._ensure_loaded()
        req_kwargs = self._with_generation_defaults(kwargs)
        req = GenerationRequest(prompt=prompt, stream=True, **req_kwargs)
        return self._backend.stream(req)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        return self._kv_cache.stats()

    def info(self) -> dict:
        return {
            "model": self.cfg.model_name,
            "backend": self.cfg.backend.value,
            "quantization": self.cfg.quant.mode.value,
            "assistant_model": self.cfg.assistant_model_name,
            "device": self.cfg.device,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "pipeline_parallel_size": self.cfg.pipeline_parallel_size,
            "max_batch_size": self.cfg.max_batch_size,
            "batch_timeout_ms": self.cfg.batch_timeout_ms,
            "paged_kv_enabled": self.cfg.cache.enable_paged_kv,
            "hf_revision": self.cfg.hf_revision,
            "hf_local_files_only": self.cfg.hf_local_files_only,
            "compile_fullgraph": self.cfg.compile_fullgraph,
            "compile_cudagraphs": self.cfg.compile_cudagraphs,
            "speculative_num_assistant_tokens": self.cfg.speculative_num_assistant_tokens,
            "speculative_confidence_threshold": self.cfg.speculative_confidence_threshold,
            "loaded": self._backend is not None and self._backend.is_loaded,
        }

    def load_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        self._ensure_loaded()
        self._backend.load_adapter(adapter_path=adapter_path, adapter_name=adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        self._ensure_loaded()
        self._backend.set_adapter(adapter_name)

    def unload_adapter(self, adapter_name: Optional[str] = None) -> None:
        self._ensure_loaded()
        self._backend.unload_adapter(adapter_name=adapter_name)

    def list_adapters(self) -> List[str]:
        self._ensure_loaded()
        return self._backend.list_adapters()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "InferenceEngine":
        return self.load()

    def __exit__(self, *args) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._backend is None or not self._backend.is_loaded:
            logger.info("Auto-loading model...")
            self.load()

    def _with_generation_defaults(self, kwargs: dict) -> dict:
        merged = dict(kwargs)
        merged.setdefault("max_new_tokens", self.cfg.max_new_tokens)
        merged.setdefault("temperature", self.cfg.temperature)
        merged.setdefault("top_p", self.cfg.top_p)
        merged.setdefault("top_k", self.cfg.top_k)
        merged.setdefault("repetition_penalty", self.cfg.repetition_penalty)
        if self.cfg.speculative_num_assistant_tokens is not None:
            merged.setdefault(
                "speculative_num_assistant_tokens",
                self.cfg.speculative_num_assistant_tokens,
            )
        if self.cfg.speculative_confidence_threshold is not None:
            merged.setdefault(
                "speculative_confidence_threshold",
                self.cfg.speculative_confidence_threshold,
            )
        return merged

    @staticmethod
    def _run_async(coro):
        """
        Run an async coroutine from sync contexts, including notebook/event-loop
        environments by hopping to a worker thread when needed.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        holder: Dict[str, object] = {"result": None, "error": None}

        def _runner() -> None:
            try:
                holder["result"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - passthrough
                holder["error"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if holder["error"] is not None:
            raise holder["error"]  # type: ignore[misc]
        return holder["result"]
