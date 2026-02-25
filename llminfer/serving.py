"""
Serving primitives for llminfer.

Provides an async continuous-batching scheduler that can sit in front of
InferenceEngine for production-style request coalescing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

from llminfer.engine import InferenceEngine
from llminfer.request import GenerationRequest, GenerationResult


@dataclass
class QueuedRequest:
    request: GenerationRequest
    future: asyncio.Future


def chat_messages_to_prompt(messages: List[dict]) -> str:
    """Convert OpenAI-style chat messages into a plain text prompt."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role_prefix = role.upper()
        lines.append(f"{role_prefix}: {content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler for heterogeneous GenerationRequest objects.

    Requests are grouped for up to `batch_timeout_ms` or until `max_batch_size`
    is reached, then executed as a single backend batch.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int,
        batch_timeout_ms: float,
        max_queue_size: int = 1024,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._stop_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return

        await asyncio.to_thread(self.engine.load)
        self._stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="llminfer-batch-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self._worker_task is None or self._worker_task.done():
            await self.start()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(QueuedRequest(request=request, future=fut))
        return await fut

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break

            batch = [first]
            deadline = time.monotonic() + self.batch_timeout_s

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            requests = [item.request for item in batch]
            try:
                results = await asyncio.to_thread(self.engine.run_requests, requests)
                result_map = {r.request_id: r for r in results}
                for item in batch:
                    if item.future.done():
                        continue
                    result = result_map.get(item.request.request_id)
                    if result is None:
                        item.future.set_exception(
                            RuntimeError(f"Missing result for request {item.request.request_id}")
                        )
                    else:
                        item.future.set_result(result)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
