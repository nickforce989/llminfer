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

import logging
from typing import Dict, Iterator, List, Optional

from llminfer.backends.base import BaseBackend
from llminfer.backends.compiled import CompiledBackend
from llminfer.backends.eager import EagerBackend
from llminfer.batching import SyncBatchQueue
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
        req = GenerationRequest(prompt=prompt, **kwargs)
        results = self._backend.generate([req])
        return results[0]

    def run_batch(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """
        Generate completions for a list of prompts.
        Automatically splits into batches of max_batch_size.
        """
        self._ensure_loaded()
        requests = [GenerationRequest(prompt=p, **kwargs) for p in prompts]
        results: List[GenerationResult] = []

        # Process in chunks of max_batch_size
        bs = self.cfg.max_batch_size
        for i in range(0, len(requests), bs):
            chunk = requests[i : i + bs]
            results.extend(self._backend.generate(chunk))

        return results

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
        req = GenerationRequest(prompt=prompt, stream=True, **kwargs)
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
            "device": self.cfg.device,
            "max_batch_size": self.cfg.max_batch_size,
            "loaded": self._backend is not None and self._backend.is_loaded,
        }

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
