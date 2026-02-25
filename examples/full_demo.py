"""
examples/full_demo.py

Demonstrates all llminfer features:
  1. Basic inference (eager, no quant)
  2. 4-bit quantized inference
  3. Streaming output
  4. Batch inference
  5. KV cache prefix reuse
  6. Throughput benchmark
  7. Backend comparison (eager vs compiled)
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

from llminfer import InferenceEngine, EngineConfig, Benchmarker
from llminfer.benchmark import BackendComparison
from llminfer.config import Backend, CacheConfig, QuantConfig, QuantMode
from llminfer.kv_cache import KVCacheManager

MODEL = "facebook/opt-125m"   # swap for "meta-llama/Llama-2-7b-hf" etc.

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic eager inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. Basic eager inference")
print("="*60)

engine = InferenceEngine(EngineConfig(model_name=MODEL))
engine.load()

result = engine.run("What is a transformer neural network?", max_new_tokens=64)
print("Prompt   :", result.prompt)
print("Response :", result.generated_text)
print(f"Latency  : {result.stats.total_latency_ms:.0f} ms")
print(f"Tok/sec  : {result.stats.throughput_tokens_per_sec:.1f}")
engine.unload()

# ─────────────────────────────────────────────────────────────────────────────
# 2. 4-bit (NF4) quantized inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. 4-bit NF4 quantized inference")
print("="*60)

cfg_nf4 = EngineConfig(
    model_name=MODEL,
    quant=QuantConfig(mode=QuantMode.NF4, double_quant=True, compute_dtype="bfloat16"),
)
engine_nf4 = InferenceEngine(cfg_nf4)
engine_nf4.load()

result = engine_nf4.run("Explain attention mechanisms in one paragraph.", max_new_tokens=80)
print("Response:", result.generated_text[:300])
print(f"GPU mem (quantized): approx {result.stats.prompt_tokens} prompt tokens handled")
engine_nf4.unload()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Token streaming
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. Token streaming")
print("="*60)

engine = InferenceEngine(EngineConfig(model_name=MODEL))
engine.load()

print("Streaming: ", end="")
for chunk in engine.stream("Once upon a time in a GPU cluster far away,", max_new_tokens=48):
    if not chunk.is_final:
        print(chunk.token, end="", flush=True)
    else:
        print()
        if chunk.stats:
            print(f"TTFT: {chunk.stats.time_to_first_token_ms:.0f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. Batch inference (batch_size=4)")
print("="*60)

prompts = [
    "What is gradient descent?",
    "Explain the vanishing gradient problem.",
    "What is batch normalization?",
    "Describe the BERT architecture briefly.",
]
results = engine.run_batch(prompts, max_new_tokens=48)
for r in results:
    print(f"  [{r.request_id}] {r.generated_text[:80]}...")

# ─────────────────────────────────────────────────────────────────────────────
# 5. KV cache prefix reuse
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. KV cache prefix reuse")
print("="*60)

system_prompt = (
    "You are a helpful AI assistant specializing in machine learning. "
    "Answer questions concisely and accurately."
)
prefix_key = KVCacheManager.hash_prefix(system_prompt)

# First call: cache miss, builds KV for system prompt
r1 = engine.run(
    system_prompt + "\n\nUser: What is dropout?",
    max_new_tokens=48,
    prefix_key=prefix_key,
)
print(f"First call  cache_hit={r1.stats.cache_hit}  latency={r1.stats.total_latency_ms:.0f}ms")

# Second call: cache hit (same prefix key) → faster
r2 = engine.run(
    system_prompt + "\n\nUser: What is layer normalization?",
    max_new_tokens=48,
    prefix_key=prefix_key,
)
print(f"Second call cache_hit={r2.stats.cache_hit}  latency={r2.stats.total_latency_ms:.0f}ms")
print("Cache stats:", engine.cache_stats())
engine.unload()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Throughput + latency benchmark
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. Throughput & latency benchmark")
print("="*60)

engine = InferenceEngine(EngineConfig(model_name=MODEL, max_batch_size=8))
engine.load()

bm = Benchmarker(engine)
bench_result = bm.run(
    batch_sizes=[1, 2, 4, 8],
    num_runs=5,
    warmup_runs=2,
    max_new_tokens=64,
)
bench_result.print_summary()
bench_result.plot("benchmark_eager.png")
bench_result.plot_suite("benchmark_eager_plots", prefix="benchmark_eager")
engine.unload()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Backend comparison: eager vs torch.compile
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. Backend comparison: eager vs torch.compile")
print("="*60)

cmp = BackendComparison(
    model_name=MODEL,
    backends=[Backend.EAGER, Backend.COMPILED],
)
results = cmp.run(batch_sizes=[1, 2, 4, 8], num_runs=5)
cmp.print_table(results)
cmp.plot(results, "comparison_eager_vs_compiled.png")
cmp.plot_suite(results, "comparison_plots", prefix="comparison_eager_vs_compiled")

print("\n✅ Done. Check benchmark_eager.png, benchmark_eager_plots/, comparison_eager_vs_compiled.png, and comparison_plots/")
