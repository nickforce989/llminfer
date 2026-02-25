"""
KV Cache manager.

Responsibilities
----------------
1. Store past_key_values per sequence slot.
2. Prefix caching: hash common prompt prefixes and reuse their KV blocks.
3. Eviction: FIFO or LRU when the cache is full.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from llminfer.config import CacheConfig, CacheEviction

logger = logging.getLogger(__name__)

# past_key_values is a tuple-of-tuples: ((k, v), (k, v), ...) one per layer
from typing import Any
PastKV = Tuple[Tuple[Any, Any], ...]


@dataclass
class CacheEntry:
    past_kv: PastKV
    seq_len: int          # Number of tokens whose KV is stored
    last_used: float = field(default_factory=time.monotonic)
    hits: int = 0


class KVCacheManager:
    """
    Manages KV cache slots for active sequences and prefix-cached prompts.

    Usage
    -----
    cache = KVCacheManager(cfg)
    past_kv, hit = cache.get_prefix(prefix_key, prompt_tokens)
    ... run forward pass with past_kv ...
    cache.update(request_id, new_past_kv)
    cache.evict_if_needed()
    """

    def __init__(self, cfg: CacheConfig) -> None:
        self.cfg = cfg
        # Active sequence cache: seq_id -> CacheEntry
        self._active: OrderedDict[str, CacheEntry] = OrderedDict()
        # Prefix cache: hash -> CacheEntry
        self._prefix: OrderedDict[str, CacheEntry] = OrderedDict()

        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Sequence-level cache (per active request)
    # ------------------------------------------------------------------

    def has(self, seq_id: str) -> bool:
        return seq_id in self._active

    def get(self, seq_id: str) -> Optional[PastKV]:
        entry = self._active.get(seq_id)
        if entry is None:
            return None
        entry.last_used = time.monotonic()
        entry.hits += 1
        # Move to end (LRU touch)
        self._active.move_to_end(seq_id)
        return entry.past_kv

    def update(self, seq_id: str, past_kv: PastKV, seq_len: int) -> None:
        self._active[seq_id] = CacheEntry(past_kv=past_kv, seq_len=seq_len)
        self._active.move_to_end(seq_id)
        self._maybe_evict_active()

    def free(self, seq_id: str) -> None:
        self._active.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Prefix cache
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prefix(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def lookup_prefix(self, key: str) -> Optional[PastKV]:
        """Return cached KV for a prefix key (e.g. system prompt hash)."""
        if not self.cfg.enable_prefix_cache:
            return None
        entry = self._prefix.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        entry.last_used = time.monotonic()
        entry.hits += 1
        self._prefix.move_to_end(key)
        logger.debug("Prefix cache HIT  key=%s hits=%d", key, entry.hits)
        return entry.past_kv

    def store_prefix(self, key: str, past_kv: PastKV, seq_len: int) -> None:
        if not self.cfg.enable_prefix_cache:
            return
        self._prefix[key] = CacheEntry(past_kv=past_kv, seq_len=seq_len)
        self._prefix.move_to_end(key)
        self._maybe_evict_prefix()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_evict_active(self) -> None:
        while len(self._active) > self.cfg.max_seqs:
            if self.cfg.eviction == CacheEviction.LRU:
                evicted_id, _ = self._active.popitem(last=False)
            elif self.cfg.eviction == CacheEviction.FIFO:
                evicted_id, _ = next(iter(self._active.items()))
                del self._active[evicted_id]
            else:
                break
            logger.debug("Evicted active seq %s", evicted_id)

    def _maybe_evict_prefix(self) -> None:
        # Keep at most max_seqs/2 prefix entries
        limit = max(1, self.cfg.max_seqs // 2)
        while len(self._prefix) > limit:
            evicted_key, _ = self._prefix.popitem(last=False)
            logger.debug("Evicted prefix key %s", evicted_key)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "active_seqs": len(self._active),
            "prefix_entries": len(self._prefix),
            "prefix_hit_rate": f"{self.hit_rate:.2%}",
            "total_hits": self._hits,
            "total_misses": self._misses,
        }
