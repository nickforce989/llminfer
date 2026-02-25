"""
KV Cache manager.

Responsibilities
----------------
1. Store past_key_values per sequence slot.
2. Prefix caching: hash common prompt prefixes and reuse their KV blocks.
3. Optional paged KV cache representation (fixed-size token pages).
4. Eviction: FIFO or LRU when the cache is full.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llminfer.config import CacheConfig, CacheEviction

logger = logging.getLogger(__name__)

# past_key_values is a tuple-of-tuples: ((k, v), (k, v), ...) one per layer
PastKV = Tuple[Tuple[Any, Any], ...]


@dataclass
class CacheEntry:
    # Either a monolithic `past_kv` or a paged representation in `pages`.
    past_kv: Optional[PastKV]
    seq_len: int          # Number of tokens whose KV is stored
    pages: Tuple[PastKV, ...] = field(default_factory=tuple)
    paged: bool = False
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
        self._paged_reads = 0

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
        return self._materialize_entry(entry)

    def update(self, seq_id: str, past_kv: PastKV, seq_len: int) -> None:
        self._active[seq_id] = self._build_entry(past_kv=past_kv, seq_len=seq_len)
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
        return self._materialize_entry(entry)

    def store_prefix(self, key: str, past_kv: PastKV, seq_len: int) -> None:
        if not self.cfg.enable_prefix_cache:
            return
        self._prefix[key] = self._build_entry(past_kv=past_kv, seq_len=seq_len)
        self._prefix.move_to_end(key)
        self._maybe_evict_prefix()

    # ------------------------------------------------------------------
    # Paged representation
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_seq_len_from_kv(past_kv: PastKV) -> int:
        if not past_kv:
            return 0
        for layer in past_kv:
            if not isinstance(layer, (tuple, list)) or len(layer) < 1:
                continue
            key_tensor = layer[0]
            shape = getattr(key_tensor, "shape", None)
            if shape is None or len(shape) < 2:
                continue
            # HF cache convention is [..., seq, head_dim], so seq axis is -2.
            return int(shape[-2]) if len(shape) >= 2 else 0
        return 0

    @staticmethod
    def _can_page(past_kv: PastKV) -> bool:
        if not past_kv:
            return False
        for layer in past_kv:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                return False
            k, v = layer[0], layer[1]
            if getattr(k, "ndim", 0) < 3 or getattr(v, "ndim", 0) < 3:
                return False
        return True

    @staticmethod
    def _slice_seq_dim(tensor: Any, start: int, end: int) -> Any:
        ndim = getattr(tensor, "ndim", 0)
        if ndim < 3:
            return tensor
        dim = ndim - 2
        if hasattr(tensor, "narrow"):
            # clone() avoids sharing mutable buffers between page slices.
            return tensor.narrow(dim, start, end - start).clone()
        slicer = [slice(None)] * ndim
        slicer[dim] = slice(start, end)
        return tensor[tuple(slicer)]

    @classmethod
    def _split_into_pages(cls, past_kv: PastKV, seq_len: int, page_size: int) -> Tuple[PastKV, ...]:
        pages: List[PastKV] = []
        for start in range(0, seq_len, page_size):
            end = min(start + page_size, seq_len)
            page_layers: List[Tuple[Any, Any]] = []
            for layer in past_kv:
                k, v = layer[0], layer[1]
                page_layers.append(
                    (
                        cls._slice_seq_dim(k, start, end),
                        cls._slice_seq_dim(v, start, end),
                    )
                )
            pages.append(tuple(page_layers))
        return tuple(pages)

    @staticmethod
    def _concat_seq_dim(chunks: List[Any]) -> Any:
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        sample = chunks[0]
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency here
            return chunks[-1]
        if isinstance(sample, torch.Tensor):
            dim = sample.ndim - 2 if sample.ndim >= 3 else 0
            return torch.cat(chunks, dim=dim)
        return chunks[-1]

    @classmethod
    def _merge_pages(cls, pages: Tuple[PastKV, ...]) -> Optional[PastKV]:
        if not pages:
            return None
        num_layers = len(pages[0])
        merged_layers: List[Tuple[Any, Any]] = []
        for layer_idx in range(num_layers):
            layer0 = pages[0][layer_idx]
            if not isinstance(layer0, (tuple, list)) or len(layer0) < 2:
                # Fallback path for non-standard cache structures.
                merged_layers.append(layer0)  # type: ignore[arg-type]
                continue
            keys = [page[layer_idx][0] for page in pages]
            vals = [page[layer_idx][1] for page in pages]
            merged_layers.append(
                (
                    cls._concat_seq_dim(keys),
                    cls._concat_seq_dim(vals),
                )
            )
        return tuple(merged_layers)

    def _build_entry(self, past_kv: PastKV, seq_len: int) -> CacheEntry:
        effective_seq_len = int(seq_len) if seq_len > 0 else self._infer_seq_len_from_kv(past_kv)
        page_size = max(1, int(self.cfg.page_size_tokens))
        if (
            self.cfg.enable_paged_kv
            and effective_seq_len > page_size
            and self._can_page(past_kv)
        ):
            pages = self._split_into_pages(
                past_kv=past_kv,
                seq_len=effective_seq_len,
                page_size=page_size,
            )
            return CacheEntry(
                past_kv=None,
                pages=pages,
                seq_len=effective_seq_len,
                paged=True,
            )
        return CacheEntry(
            past_kv=past_kv,
            pages=tuple(),
            seq_len=effective_seq_len,
            paged=False,
        )

    def _materialize_entry(self, entry: CacheEntry) -> Optional[PastKV]:
        if entry.paged:
            self._paged_reads += 1
            return self._merge_pages(entry.pages)
        return entry.past_kv

    @staticmethod
    def _entry_pages(entry: CacheEntry) -> int:
        if entry.paged:
            return len(entry.pages)
        return 1 if entry.past_kv is not None else 0

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
        active_paged_entries = sum(1 for e in self._active.values() if e.paged)
        prefix_paged_entries = sum(1 for e in self._prefix.values() if e.paged)
        active_pages = sum(self._entry_pages(e) for e in self._active.values())
        prefix_pages = sum(self._entry_pages(e) for e in self._prefix.values())
        return {
            "active_seqs": len(self._active),
            "prefix_entries": len(self._prefix),
            "prefix_hit_rate": f"{self.hit_rate:.2%}",
            "total_hits": self._hits,
            "total_misses": self._misses,
            "paged_kv_enabled": self.cfg.enable_paged_kv,
            "page_size_tokens": self.cfg.page_size_tokens,
            "active_paged_entries": active_paged_entries,
            "prefix_paged_entries": prefix_paged_entries,
            "active_pages": active_pages,
            "prefix_pages": prefix_pages,
            "paged_reads": self._paged_reads,
        }
