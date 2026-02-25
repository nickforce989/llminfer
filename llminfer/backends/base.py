"""
Abstract base class for inference backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from llminfer.config import EngineConfig
from llminfer.request import GenerationRequest, GenerationResult, StreamChunk


class BaseBackend(ABC):
    """
    All backends must implement `generate` (batch) and `stream` (single-req).
    The engine selects the appropriate backend based on EngineConfig.backend.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load model + tokenizer onto device."""
        ...

    @abstractmethod
    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Run batch inference synchronously.
        Returns one GenerationResult per request, in order.
        """
        ...

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """
        Stream tokens for a single request.
        Yields StreamChunk objects, final chunk has is_final=True.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        import torch
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optional adapter API (implemented by backends that support PEFT)
    # ------------------------------------------------------------------

    def load_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support adapters")

    def set_adapter(self, adapter_name: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support adapters")

    def unload_adapter(self, adapter_name: Optional[str] = None) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support adapters")

    def list_adapters(self) -> List[str]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support adapters")
