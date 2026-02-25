"""
torch.compile backend.

Wraps the EagerBackend but compiles the model's forward pass
with torch.compile() for significant throughput improvements on
repeated shapes (especially with reduce-overhead or max-autotune modes).

Key differences from eager:
  - First call is slower (compilation), subsequent calls are faster.
  - Best gains with static shapes (fixed batch size, fixed seq len).
  - dynamic=True allows variable shapes at the cost of retracing.

Measured speedups (A100, LLaMA-7B):
  reduce-overhead:  ~1.3–1.8× throughput vs eager
  max-autotune:     ~1.5–2.2× throughput vs eager (much longer warmup)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, List

import torch

from llminfer.backends.eager import EagerBackend
from llminfer.config import EngineConfig
from llminfer.request import GenerationRequest, GenerationResult, StreamChunk

logger = logging.getLogger(__name__)


class CompiledBackend(EagerBackend):
    """
    PyTorch torch.compile backend.

    Inherits all load / generate / stream logic from EagerBackend,
    but compiles the model's forward method after loading.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__(cfg)
        self._compiled = False
        self._warmup_done = False
        self._original_forward = None

    def load(self) -> None:
        super().load()
        self._compile_model()

    def _compile_model(self) -> None:
        if not torch.cuda.is_available():
            logger.warning(
                "torch.compile requires CUDA; falling back to eager on CPU."
            )
            return

        self._configure_inductor()

        logger.info(
            "Compiling model  mode=%s  dynamic=%s  fullgraph=%s  cudagraphs=%s",
            self.cfg.compile_mode,
            self.cfg.compile_dynamic,
            self.cfg.compile_fullgraph,
            self.cfg.compile_cudagraphs,
        )
        t0 = time.monotonic()
        self._original_forward = self._model.forward

        # Compile the forward pass only (not the entire generate loop)
        try:
            self._model.forward = torch.compile(
                self._model.forward,
                mode=self.cfg.compile_mode,
                dynamic=self.cfg.compile_dynamic,
                fullgraph=self.cfg.compile_fullgraph,
            )
        except Exception:
            logger.exception("torch.compile setup failed; using eager forward path.")
            self._compiled = False
            return
        self._compiled = True

        logger.info("Compilation setup done in %.2fs  (first call will trigger JIT)", time.monotonic() - t0)

    def _configure_inductor(self) -> None:
        """
        Configure inductor cudagraph behavior, if exposed by this torch build.
        """
        try:
            import torch._inductor.config as inductor_config
        except Exception:
            return
        try:
            triton_cfg = getattr(inductor_config, "triton", None)
            if triton_cfg is None:
                return
            if hasattr(triton_cfg, "cudagraphs"):
                triton_cfg.cudagraphs = bool(self.cfg.compile_cudagraphs)
            if hasattr(triton_cfg, "cudagraph_trees"):
                triton_cfg.cudagraph_trees = bool(self.cfg.compile_cudagraphs)
        except Exception:
            logger.debug("Could not apply inductor cudagraph config", exc_info=True)

    def _mark_cudagraph_step(self) -> None:
        """
        Mark a new cudagraph step boundary before each compiled invocation.

        This avoids output-buffer reuse issues seen on some torch/transformers
        combinations in long generation loops.
        """
        if not self._compiled:
            return
        try:
            mark_step = torch.compiler.cudagraph_mark_step_begin
        except AttributeError:
            return
        mark_step()

    def warmup(self, prompt: str = "Hello world", n: int = 2) -> None:
        """
        Run a few dummy generations to trigger JIT compilation
        and populate the CUDA graph cache before real traffic.
        """
        if not self._compiled:
            return
        logger.info("Warming up compiled model (%d passes)...", n)
        t0 = time.monotonic()
        req = GenerationRequest(prompt=prompt, max_new_tokens=16)
        for _ in range(n):
            self.generate([req])
        self._warmup_done = True
        logger.info("Warmup complete in %.2fs", time.monotonic() - t0)

    @staticmethod
    def _looks_like_compile_runtime_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        hints = (
            "cudagraph",
            "torch._dynamo",
            "aot_autograd",
            "torch._inductor",
            "inductor",
            "overwritten by a subsequent run",
        )
        return any(h in msg for h in hints)

    def _fallback_to_eager(self, exc: Exception) -> None:
        if not self.cfg.compile_fallback_to_eager:
            raise exc
        logger.warning("Compiled backend failed at runtime; falling back to eager: %s", exc)
        if self._original_forward is not None and self._model is not None:
            self._model.forward = self._original_forward
        self._compiled = False
        self._warmup_done = False

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        self._mark_cudagraph_step()
        try:
            results = super().generate(requests)
        except RuntimeError as exc:
            if self._compiled and self._looks_like_compile_runtime_error(exc):
                self._fallback_to_eager(exc)
                results = super().generate(requests)
            else:
                raise
        if self._compiled and not self._warmup_done:
            logger.debug("First compiled inference — JIT tracing occurred.")
        return results

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        # Streaming with torch.compile works the same way; the compiled
        # forward will be called inside the generate loop.
        self._mark_cudagraph_step()
        parent_stream = super(CompiledBackend, self).stream

        def _iterator() -> Iterator[StreamChunk]:
            try:
                for chunk in parent_stream(request):
                    yield chunk
            except RuntimeError as exc:
                if self._compiled and self._looks_like_compile_runtime_error(exc):
                    self._fallback_to_eager(exc)
                    for chunk in parent_stream(request):
                        yield chunk
                else:
                    raise

        return _iterator()
