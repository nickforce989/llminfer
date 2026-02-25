"""
tests/test_core.py — unit tests that run without a GPU or model download.
"""

import pytest
from unittest.mock import MagicMock, patch
import time


# ─── Config ────────────────────────────────────────────────────────────────

def test_quant_config_no_quant():
    from llminfer.config import QuantConfig, QuantMode
    cfg = QuantConfig(mode=QuantMode.NONE)
    assert cfg.to_bnb_kwargs() == {}


def test_quant_config_int8():
    from llminfer.config import QuantConfig, QuantMode
    cfg = QuantConfig(mode=QuantMode.INT8)
    kwargs = cfg.to_bnb_kwargs()
    assert kwargs.get("load_in_8bit") is True


def test_quant_config_nf4():
    from llminfer.config import QuantConfig, QuantMode
    cfg = QuantConfig(mode=QuantMode.NF4, double_quant=True)
    kwargs = cfg.to_bnb_kwargs()
    assert kwargs.get("load_in_4bit") is True
    assert kwargs.get("bnb_4bit_quant_type") == "nf4"
    assert kwargs.get("bnb_4bit_use_double_quant") is True


def test_engine_config_defaults():
    from llminfer.config import EngineConfig, Backend, QuantMode
    cfg = EngineConfig()
    assert cfg.backend == Backend.EAGER
    assert cfg.quant.mode == QuantMode.NONE
    assert cfg.max_batch_size == 8


# ─── KV Cache ───────────────────────────────────────────────────────────────

def test_kv_cache_basic():
    from llminfer.config import CacheConfig
    from llminfer.kv_cache import KVCacheManager
    import torch

    cfg = CacheConfig(max_seqs=3, enable_prefix_cache=True)
    cache = KVCacheManager(cfg)

    # Fake past_kv tuple
    fake_kv = ((torch.zeros(1, 4, 8, 16), torch.zeros(1, 4, 8, 16)),) * 6

    cache.update("seq1", fake_kv, seq_len=10)
    assert cache.has("seq1")
    assert cache.get("seq1") is not None

    cache.free("seq1")
    assert not cache.has("seq1")


def test_kv_cache_lru_eviction():
    from llminfer.config import CacheConfig, CacheEviction
    from llminfer.kv_cache import KVCacheManager
    import torch

    cfg = CacheConfig(max_seqs=2, eviction=CacheEviction.LRU)
    cache = KVCacheManager(cfg)
    fake_kv = ((torch.zeros(1, 1, 4, 8),),) * 2

    cache.update("a", fake_kv, 5)
    cache.update("b", fake_kv, 5)
    cache.update("c", fake_kv, 5)  # should evict 'a'

    assert not cache.has("a"), "LRU should have evicted 'a'"
    assert cache.has("b")
    assert cache.has("c")


def test_prefix_cache_hit():
    from llminfer.config import CacheConfig
    from llminfer.kv_cache import KVCacheManager
    import torch

    cfg = CacheConfig(enable_prefix_cache=True)
    cache = KVCacheManager(cfg)
    fake_kv = ((torch.zeros(1, 1, 4, 8),),) * 2

    key = KVCacheManager.hash_prefix("system: you are helpful")
    assert cache.lookup_prefix(key) is None  # miss

    cache.store_prefix(key, fake_kv, 10)
    result = cache.lookup_prefix(key)
    assert result is not None  # hit
    assert cache.hit_rate > 0


def test_prefix_cache_disabled():
    from llminfer.config import CacheConfig
    from llminfer.kv_cache import KVCacheManager
    import torch

    cfg = CacheConfig(enable_prefix_cache=False)
    cache = KVCacheManager(cfg)
    fake_kv = ((torch.zeros(1, 1, 4, 8),),) * 2

    key = "some_key"
    cache.store_prefix(key, fake_kv, 10)
    assert cache.lookup_prefix(key) is None  # disabled


# ─── Batching ───────────────────────────────────────────────────────────────

def test_sync_batch_queue():
    from llminfer.batching import SyncBatchQueue
    from llminfer.request import GenerationRequest

    q = SyncBatchQueue(max_batch_size=3)
    for i in range(5):
        q.add(GenerationRequest(prompt=f"prompt {i}"))

    batches = q.flush_all()
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2


def test_sync_batch_queue_empty():
    from llminfer.batching import SyncBatchQueue
    q = SyncBatchQueue(max_batch_size=4)
    assert q.flush() is None


# ─── Request ────────────────────────────────────────────────────────────────

def test_generation_request_defaults():
    from llminfer.request import GenerationRequest
    req = GenerationRequest(prompt="hello")
    assert req.max_new_tokens == 256
    assert req.temperature == 1.0
    assert req.request_id is not None
    assert req.arrival_time > 0


def test_token_stats_tokens_per_second():
    from llminfer.request import TokenStats
    stats = TokenStats(generated_tokens=100, total_latency_ms=1000)
    assert abs(stats.tokens_per_second - 100.0) < 0.01


# ─── Benchmark result ───────────────────────────────────────────────────────

def test_run_metrics():
    from llminfer.benchmark import RunMetrics
    m = RunMetrics(batch_size=4, prompt_type="short")
    m.latencies_ms = [100, 110, 90, 200, 95]
    m.throughputs_tps = [300, 320, 280, 250, 310]

    assert m.mean_latency_ms == pytest.approx(119.0)
    assert m.p95_latency_ms >= m.p50_latency_ms
    assert m.mean_throughput_tps > 0


def test_benchmark_result_curves():
    from llminfer.benchmark import BenchmarkResult, RunMetrics
    result = BenchmarkResult(backend_name="eager", model_name="test", quant_mode="none")
    for bs in [1, 2, 4]:
        m = RunMetrics(batch_size=bs, prompt_type="medium")
        m.latencies_ms = [100.0 * bs]
        m.throughputs_tps = [50.0 * bs]
        result.metrics_by_batch[bs] = m

    xs, tps = result.throughput_curve
    assert xs == [1, 2, 4]
    assert tps[2] > tps[0]   # throughput should increase with batch size

    xs, lat = result.latency_curve
    assert lat[2] > lat[0]   # latency should increase with batch size


# ─── Engine config / backend selection ─────────────────────────────────────

def test_engine_build_eager_backend():
    from llminfer.engine import InferenceEngine
    from llminfer.config import EngineConfig, Backend
    from llminfer.backends.eager import EagerBackend

    cfg = EngineConfig(backend=Backend.EAGER)
    engine = InferenceEngine(cfg)
    backend = engine._build_backend()
    assert isinstance(backend, EagerBackend)


def test_engine_build_compiled_backend():
    from llminfer.engine import InferenceEngine
    from llminfer.config import EngineConfig, Backend
    from llminfer.backends.compiled import CompiledBackend

    cfg = EngineConfig(backend=Backend.COMPILED)
    engine = InferenceEngine(cfg)
    backend = engine._build_backend()
    assert isinstance(backend, CompiledBackend)


def test_engine_info_not_loaded():
    from llminfer.engine import InferenceEngine
    from llminfer.config import EngineConfig
    engine = InferenceEngine(EngineConfig())
    info = engine.info()
    assert info["loaded"] is False
    assert "model" in info


def test_engine_context_manager_mocked():
    """Verify context manager calls load/unload without actual model."""
    from llminfer.engine import InferenceEngine
    from llminfer.config import EngineConfig

    engine = InferenceEngine(EngineConfig())
    engine.load = MagicMock(return_value=engine)
    engine.unload = MagicMock()

    with engine as e:
        assert e is engine

    engine.load.assert_called_once()
    engine.unload.assert_called_once()
