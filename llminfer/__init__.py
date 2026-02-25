"""
llminfer — GPU-Efficient LLM Inference Engine

Features:
  - Quantized weights (4-bit, 8-bit via bitsandbytes)
  - Static + continuous batching and token streaming
  - KV cache reuse with optional paged KV representation
  - Tensor/pipeline parallel controls (backend dependent)
  - Speculative decoding controls with assistant models
  - Throughput / latency benchmarking
  - Backend comparison: eager | torch.compile | vllm
"""

from llminfer.engine import InferenceEngine
from llminfer.config import EngineConfig, QuantConfig, CacheConfig
from llminfer.benchmark import Benchmarker, BenchmarkResult
from llminfer.request import GenerationRequest, GenerationResult
from llminfer.serving import ContinuousBatchScheduler

__version__ = "0.1.0"
__all__ = [
    "InferenceEngine",
    "EngineConfig",
    "QuantConfig",
    "CacheConfig",
    "Benchmarker",
    "BenchmarkResult",
    "GenerationRequest",
    "GenerationResult",
    "ContinuousBatchScheduler",
]
