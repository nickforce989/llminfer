"""
Streaming helpers for llminfer.

Wraps the HuggingFace TextIteratorStreamer to yield StreamChunk objects,
and provides a simple synchronous generator interface.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Generator, Iterator, List, Optional

from transformers import PreTrainedTokenizerBase, TextIteratorStreamer

from llminfer.request import GenerationRequest, StreamChunk, TokenStats


class TokenStreamer:
    """
    Wraps HuggingFace's TextIteratorStreamer and yields StreamChunk objects.

    Usage
    -----
    streamer = TokenStreamer(tokenizer, request)
    # Pass streamer.hf_streamer to model.generate(streamer=...)
    for chunk in streamer:
        print(chunk.token, end="", flush=True)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        request: GenerationRequest,
        skip_prompt: bool = True,
    ) -> None:
        self.request = request
        self.tokenizer = tokenizer
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._token_count = 0

        # HuggingFace streamer — model.generate writes tokens here
        self.hf_streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=True,
        )

    def __iter__(self) -> Generator[StreamChunk, None, None]:
        self._start_time = time.monotonic()
        generated_parts: List[str] = []

        for text_piece in self.hf_streamer:
            now = time.monotonic()
            if self._first_token_time is None:
                self._first_token_time = now
            self._token_count += 1
            generated_parts.append(text_piece)

            yield StreamChunk(
                request_id=self.request.request_id,
                token=text_piece,
                token_id=-1,  # HF streamer gives decoded text, not ids
                is_final=False,
            )

        # Final chunk with stats
        end_time = time.monotonic()
        total_ms = (end_time - (self._start_time or end_time)) * 1000
        ttft_ms = (
            ((self._first_token_time or end_time) - (self._start_time or end_time)) * 1000
        )

        stats = TokenStats(
            generated_tokens=self._token_count,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            throughput_tokens_per_sec=self._token_count / max(total_ms / 1000, 1e-9),
        )
        yield StreamChunk(
            request_id=self.request.request_id,
            token="",
            token_id=-1,
            is_final=True,
            stats=stats,
        )


class MultiStreamer:
    """
    Fan-out streamer: routes tokens from a batched generate call to
    per-request TokenStreamers via individual queues.

    This is a simplified version; production systems (vLLM, TGI) use
    more sophisticated per-slot demultiplexing.
    """

    def __init__(self, request_ids: List[str]) -> None:
        self.request_ids = request_ids
        self._queues: dict[str, Queue] = {rid: Queue() for rid in request_ids}

    def put(self, request_id: str, token: str, is_final: bool = False) -> None:
        self._queues[request_id].put((token, is_final))

    def stream(self, request_id: str) -> Iterator[str]:
        q = self._queues[request_id]
        while True:
            try:
                token, is_final = q.get(timeout=30.0)
                if token:
                    yield token
                if is_final:
                    break
            except Empty:
                break
