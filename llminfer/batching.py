"""
Dynamic batching for llminfer.

The BatchQueue collects incoming GenerationRequests and groups them
into micro-batches, either when the batch is full or when the timeout
fires — whichever comes first.

This mirrors production serving systems (Triton, vLLM, TGI).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from llminfer.config import EngineConfig
from llminfer.request import GenerationRequest

logger = logging.getLogger(__name__)


class Batch:
    """A group of requests to be processed together."""

    def __init__(self, requests: List[GenerationRequest]) -> None:
        self.requests = requests
        self.created_at = time.monotonic()

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def prompts(self) -> List[str]:
        return [r.prompt for r in self.requests]

    @property
    def request_ids(self) -> List[str]:
        return [r.request_id for r in self.requests]


class BatchQueue:
    """
    Thread/async-safe queue that assembles requests into Batch objects.

    Usage (async)
    -------------
    queue = BatchQueue(cfg)
    await queue.put(request)
    batch = await queue.get_batch()   # blocks until batch ready
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.max_batch_size = cfg.max_batch_size
        self.timeout_s = cfg.batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._pending: List[GenerationRequest] = []
        self._lock = asyncio.Lock()

    async def put(self, request: GenerationRequest) -> None:
        await self._queue.put(request)
        logger.debug("Queued request %s  qsize=%d", request.request_id, self._queue.qsize())

    async def get_batch(self) -> Batch:
        """
        Block until we have at least one request, then drain up to
        max_batch_size within the timeout window.
        """
        # Block for first request
        first = await self._queue.get()
        self._pending.append(first)

        deadline = time.monotonic() + self.timeout_s
        while len(self._pending) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                self._pending.append(req)
            except asyncio.TimeoutError:
                break

        batch = Batch(list(self._pending))
        self._pending.clear()
        logger.debug(
            "Assembled batch  size=%d  wait=%.1fms",
            len(batch),
            (time.monotonic() - batch.created_at) * 1000,
        )
        return batch

    def qsize(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Synchronous helpers (for non-async callers / benchmarking)
# ---------------------------------------------------------------------------

class SyncBatchQueue:
    """
    A simple synchronous batch assembler used by the benchmarker and
    the eager/compiled backends when called without an event loop.
    """

    def __init__(self, max_batch_size: int, timeout_ms: float = 20.0) -> None:
        self.max_batch_size = max_batch_size
        self.timeout_s = timeout_ms / 1000.0
        self._pending: List[GenerationRequest] = []

    def add(self, request: GenerationRequest) -> None:
        self._pending.append(request)

    def flush(self) -> Optional[Batch]:
        if not self._pending:
            return None
        batch = Batch(list(self._pending[: self.max_batch_size]))
        self._pending = self._pending[self.max_batch_size :]
        return batch

    def flush_all(self) -> List[Batch]:
        batches = []
        while self._pending:
            b = self.flush()
            if b:
                batches.append(b)
        return batches
