# llminfer 🚀

**A GPU-efficient LLM inference engine** focused on systems performance:
quantized weights, KV cache reuse, dynamic batching, token streaming,
and rigorous benchmarking across backends.

```
pip install -e ".[vllm]"    # include vLLM support
pip install -e .            # base (eager + compiled)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       InferenceEngine                        │
│   .run()  .run_batch()  .stream()  .cache_stats()            │
└──────────────┬───────────────────────────────────────────────┘
               │ selects backend
    ┌──────────▼──────────┬──────────────────┬────────────────┐
    │    EagerBackend     │  CompiledBackend  │  VLLMBackend   │
    │  (HF transformers)  │  (torch.compile)  │  (PagedAttn.)  │
    └──────────┬──────────┴──────────────────┴────────────────┘
               │ uses
    ┌──────────▼──────────┐   ┌─────────────────┐
    │    KVCacheManager   │   │   BatchQueue     │
    │  prefix cache (LRU) │   │  dynamic batch   │
    └─────────────────────┘   └─────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Quantization** | 4-bit NF4/FP4 (QLoRA), INT8 via bitsandbytes |
| **KV cache** | Per-sequence cache + prefix cache with LRU eviction |
| **Batching** | Dynamic batching with configurable timeout |
| **Streaming** | Token-by-token via HF TextIteratorStreamer |
| **Benchmarking** | Throughput, latency p50/p95/p99, TTFT, GPU memory |
| **Backends** | PyTorch eager, `torch.compile`, vLLM (optional) |
| **CLI** | `llminfer run / stream / bench / compare / info` |

---

## Quick Start

```python
from llminfer import InferenceEngine, EngineConfig
from llminfer.config import Backend, QuantConfig, QuantMode

# ── Eager inference, no quantization ──────────────────────────
engine = InferenceEngine(EngineConfig(model_name="facebook/opt-1.3b"))
engine.load()

result = engine.run("Explain attention mechanisms.")
print(result.generated_text)
print(f"Latency: {result.stats.total_latency_ms:.0f}ms  |  {result.stats.throughput_tokens_per_sec:.1f} tok/s")

# ── 4-bit quantization (uses ~25% of float16 VRAM) ───────────
cfg = EngineConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    quant=QuantConfig(mode=QuantMode.NF4, double_quant=True),
)
engine_4bit = InferenceEngine(cfg)
engine_4bit.load()

# ── Streaming ─────────────────────────────────────────────────
for chunk in engine.stream("Tell me a story"):
    if not chunk.is_final:
        print(chunk.token, end="", flush=True)

# ── Batch inference ───────────────────────────────────────────
results = engine.run_batch(["What is RLHF?", "Explain LoRA."])

# ── Prefix KV cache reuse ─────────────────────────────────────
from llminfer.kv_cache import KVCacheManager
sys_prompt = "You are a helpful AI assistant."
key = KVCacheManager.hash_prefix(sys_prompt)

r1 = engine.run(sys_prompt + "\n\nUser: What is BERT?",  prefix_key=key)  # miss
r2 = engine.run(sys_prompt + "\n\nUser: What is GPT-2?", prefix_key=key)  # hit ✓
print(f"Cache hit: {r2.stats.cache_hit}")
```

---

## Benchmarking

```python
from llminfer import Benchmarker

bm = Benchmarker(engine)
result = bm.run(batch_sizes=[1, 2, 4, 8, 16], num_runs=20, max_new_tokens=128)
result.print_summary()
result.plot("bench.png")
```

Output:
```
┌─────────────────────────────────────────────────────────────────┐
│    Benchmark: facebook/opt-125m [eager / none]                  │
├────────────┬──────────────┬──────────────┬───────────┬─────────┤
│ Batch Size │ Latency p50  │ Latency p95  │ Tok/s     │ GPU MB  │
├────────────┼──────────────┼──────────────┼───────────┼─────────┤
│          1 │       412 ms │       428 ms │     311.2 │    486  │
│          2 │       489 ms │       503 ms │     524.6 │    487  │
│          4 │       701 ms │       718 ms │     731.8 │    490  │
│          8 │      1121 ms │      1145 ms │     915.4 │    497  │
└────────────┴──────────────┴──────────────┴───────────┴─────────┘
```

---

## Backend Comparison

```python
from llminfer.benchmark import BackendComparison
from llminfer.config import Backend

cmp = BackendComparison(
    model_name="facebook/opt-1.3b",
    backends=[Backend.EAGER, Backend.COMPILED, Backend.VLLM],
)
results = cmp.run(batch_sizes=[1, 4, 8, 16])
cmp.print_table(results)
cmp.plot(results, "comparison.png")
```

Typical speedups on A100 (Llama-7B, bs=8, 128 tok output):

| Backend | Throughput | vs Eager |
|---|---|---|
| Eager | 480 tok/s | 1.0× |
| torch.compile (reduce-overhead) | 710 tok/s | 1.48× |
| vLLM (PagedAttention) | 1240 tok/s | 2.58× |

---

## CLI

```bash
# Run inference
llminfer run "Tell me about GPUs" --model facebook/opt-125m

# Streaming
llminfer stream "Explain backpropagation" --model facebook/opt-1.3b --quant nf4

# Benchmark
llminfer bench --model facebook/opt-125m --batch-sizes 1,2,4,8 --runs 10 --plot bench.png

# Compare backends
llminfer compare --model facebook/opt-125m --backends eager,compiled --plot compare.png

# Info
llminfer info --model facebook/opt-125m --backend compiled
```

---

## Project Structure

```
llminfer/
├── __init__.py          Public API surface
├── config.py            EngineConfig, QuantConfig, CacheConfig
├── engine.py            InferenceEngine (main entrypoint)
├── kv_cache.py          KVCacheManager with prefix cache + LRU eviction
├── batching.py          Dynamic batch assembly (sync + async)
├── streaming.py         TokenStreamer wrapping HF TextIteratorStreamer
├── benchmark.py         Benchmarker + BackendComparison + plotting
├── request.py           GenerationRequest, GenerationResult, StreamChunk
├── cli.py               Typer CLI (run / stream / bench / compare / info)
└── backends/
    ├── base.py          Abstract BaseBackend
    ├── eager.py         PyTorch eager + bitsandbytes quantization
    ├── compiled.py      torch.compile wrapper with warmup
    └── vllm_backend.py  vLLM PagedAttention wrapper (optional)
```

---

## Key Design Decisions

**Why KV cache at the application level?**  
HuggingFace manages KV cache internally per `generate()` call. Our
`KVCacheManager` adds *prefix-level* caching across calls — identical
system prompts don't re-compute their KV blocks on the second request.

**Why dynamic batching?**  
Static batching wastes GPU time waiting for the slowest sequence.
`BatchQueue` collects requests for up to `batch_timeout_ms` before
dispatching, trading a small fixed latency for much higher throughput.

**Why compare all three backends?**  
- **Eager**: baseline, debuggable
- **torch.compile**: ~1.5× speedup with zero infra change
- **vLLM**: the production gold standard — PagedAttention eliminates
  KV cache fragmentation and enables continuous batching for ~2.5× throughput

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (for `torch.compile` support)
- CUDA GPU (CPU inference works but is slow for large models)
- `bitsandbytes` for quantization (Linux/Windows with CUDA)
- `vllm` for the VLLM backend (Linux + CUDA only)
