"""
Request / result types for llminfer.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class GenerationRequest:
    """A single generation request submitted to the engine."""
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    stream: bool = False
    # Optional stop strings applied after decoding.
    stop_sequences: Optional[List[str]] = None
    # Token-level constraints (best-effort; backend support varies).
    bad_words: Optional[List[str]] = None
    force_words: Optional[List[str]] = None
    prefix_allowed_tokens_fn: Optional[Callable[[int, Any], List[int]]] = None
    # Speculative decoding knobs (backend support varies).
    speculative_num_assistant_tokens: Optional[int] = None
    speculative_confidence_threshold: Optional[float] = None
    # Optional RNG seed for reproducible sampling.
    seed: Optional[int] = None
    # Attach a prefix cache key to reuse KV blocks (e.g. system prompt hash)
    prefix_key: Optional[str] = None

    # Auto-assigned
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass
class TokenStats:
    """Per-request generation statistics."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    cache_hit: bool = False         # Whether a prefix cache hit occurred

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_ms > 0:
            return self.generated_tokens / (self.total_latency_ms / 1000)
        return 0.0


@dataclass
class GenerationResult:
    """Completed generation result."""
    request_id: str
    prompt: str
    generated_text: str
    stats: TokenStats = field(default_factory=TokenStats)
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        return self.prompt + self.generated_text


@dataclass
class StreamChunk:
    """A single token streamed back from the engine."""
    request_id: str
    token: str
    token_id: int
    is_final: bool = False
    stats: Optional[TokenStats] = None   # Only populated on final chunk
