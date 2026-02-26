# llminfer: A GPU-efficient LLM inference engine

This is a python package focused on systems performance:
quantized weights, KV cache reuse, dynamic batching, token streaming,
and rigorous benchmarking across backends.

`llminfer` is for engineers who want to:
- run and compare multiple local inference backends with one API,
- tune latency/throughput trade-offs (batching, quantization, compile),
- benchmark systematically and export reproducible artifacts,
- expose models behind an OpenAI-compatible API for internal tools.

It is not intended to replace full distributed serving stacks; it is a compact,
hackable inference/benchmarking layer you can use in experiments, Colab, and
single-node production prototypes.

```bash
# Base (eager + compiled)
pip install -e .

# Optional extras
pip install -e ".[vllm]"    # vLLM backend
pip install -e ".[serve]"   # FastAPI + uvicorn + metrics
pip install -e ".[peft]"    # LoRA adapter hot-swap
```

---

## Author
Niccolò Forzano --- [github.com/nickforce989](https://github.com/nickforce989) · [nic.forz@gmail.com](mailto:nic.forz@gmail.com)

---

## Colab Notebook Guide

- Start with `examples/llminfer_all_examples_colab.ipynb` for a complete guided tour.
- Use `examples/llminfer_colab.ipynb` for a fast smoke test.
- Use `examples/llminfer_advanced_colab.ipynb` for compile/speculative/TP tuning.
- Use `examples/llminfer_serving_colab.ipynb` for OpenAI-compatible API + SSE.
- Use `examples/llminfer_readme_colab.ipynb` as a concise README companion.

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
| **KV cache** | Per-sequence cache + prefix cache; optional paged KV representation (`page_size_tokens`) |
| **Batching** | Static batch API + continuous batching scheduler with configurable timeout |
| **Streaming** | Token-by-token via HF TextIteratorStreamer |
| **Benchmarking** | Throughput, latency p50/p95/p99, TTFT, GPU memory |
| **Backends** | PyTorch eager, `torch.compile`, vLLM (optional) |
| **Parallelism** | Tensor/pipeline parallel knobs (`--tp-size`, `--pp-size`) |
| **CUDA graphs** | Compile backend exposes cudagraph toggles (`compile_cudagraphs`, step-marking) |
| **HF loading** | Revision/token/local-files-only/cache-dir/trust-remote-code controls |
| **Serving API** | OpenAI-compatible `/v1/completions` + `/v1/chat/completions` + SSE |
| **Artifacts** | Export benchmark/comparison data to JSON + CSV with environment metadata |
| **Advanced decode** | Stop sequences, no-repeat n-gram, bad/force words, optional speculative decoding |
| **Adapters** | LoRA hot-swap via PEFT (`load_adapter`, `set_adapter`, `unload_adapter`) |
| **CLI** | `llminfer run / stream / bench / compare / serve / info` |

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

## HF Weight Loading Controls

You can load a specific Hugging Face revision, use private/gated models with a
token, force local-only loading, and pin the cache directory.

```python
cfg = EngineConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    hf_revision="main",          # branch/tag/commit
    hf_token=None,               # or "hf_xxx" for gated models
    hf_local_files_only=False,   # True = do not hit network
    hf_cache_dir="./hf-cache",
    hf_trust_remote_code=True,
)
engine = InferenceEngine(cfg).load()
```

CLI:

```bash
llminfer run "Explain tensor parallelism." \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision main \
  --cache-dir ./hf-cache \
  --hf-token "$HF_TOKEN"
```

---

## Paged KV Cache And Continuous Batching

For serving-like traffic, you can:
- represent cached KV tensors in fixed-size pages (`enable_paged_kv`),
- process requests with the continuous-batching scheduler.

```python
cfg = EngineConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_batch_size=16,
    batch_timeout_ms=20,
)
cfg.cache.enable_paged_kv = True
cfg.cache.page_size_tokens = 16

engine = InferenceEngine(cfg).load()
results = engine.run_batch_continuous(
    ["Give me 3 GPU facts.", "Explain KV cache briefly."],
    max_new_tokens=96,
    temperature=0.2,
)
for r in results:
    print(r.generated_text)
```

CLI:

```bash
llminfer bench \
  --model facebook/opt-125m \
  --batch-sizes 1,2,4,8 \
  --runs 5 \
  --continuous \
  --paged-kv \
  --page-size-tokens 16
```

---

## Advanced Inference Controls

You can pass additional controls per request to reduce repetition, enforce style,
or improve reproducibility.

```python
result = engine.run(
    "Explain GPU memory hierarchy in 5 bullets.",
    max_new_tokens=160,
    temperature=0.2,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3,
    stop_sequences=["\n\nReferences:"],
    bad_words=["I'm not sure", "joking"],
    force_words=["HBM", "Tensor Core"],
    seed=42,
)
print(result.generated_text)
```

CLI equivalents:

```bash
llminfer run "Explain GPU memory hierarchy in 5 bullets." \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --temp 0.2 \
  --top-p 0.9 \
  --no-repeat-ngram-size 3 \
  --bad-words "I'm not sure,joking" \
  --force-words "HBM,Tensor Core" \
  --stop "References:" \
  --seed 42
```

---

## Speculative Decoding (Assistant Model)

`llminfer` can pass a smaller assistant model into Hugging Face generation
(`assistant_model`) to accelerate decoding on some workloads.

```python
cfg = EngineConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    assistant_model_name="Qwen/Qwen2.5-0.5B-Instruct",
)
engine = InferenceEngine(cfg).load()
print(engine.run("What is speculative decoding?", max_new_tokens=96, temperature=0.2).generated_text)
```

Advanced speculative knobs:

```python
cfg = EngineConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    assistant_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    speculative_num_assistant_tokens=8,
    speculative_confidence_threshold=0.4,
)
```

CLI:

```bash
llminfer run "What is speculative decoding?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --assistant-model Qwen/Qwen2.5-0.5B-Instruct \
  --spec-num-assistant-tokens 8 \
  --spec-confidence-threshold 0.4 \
  --temp 0.2
```

---

## Adapter Hot-Swap (PEFT LoRA)

Use adapters without reloading the base model.

```python
engine = InferenceEngine(EngineConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")).load()
engine.load_adapter("your-adapter-path-or-hf-repo", adapter_name="domain_a")
print(engine.list_adapters())
engine.set_adapter("domain_a")
print(engine.run("Summarize LoRA adapters in one paragraph.", max_new_tokens=96).generated_text)
engine.unload_adapter("domain_a")
```

CLI:

```bash
llminfer run "Summarize LoRA adapters." \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter your-adapter-path-or-hf-repo \
  --adapter-name domain_a
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

## Benchmark Artifacts (JSON / CSV)

Benchmarks and backend comparisons can be exported as machine-readable artifacts
with environment metadata (Python, PyTorch, CUDA, GPU, transformers versions).

Single backend benchmark:

```bash
llminfer bench \
  --model facebook/opt-125m \
  --batch-sizes 1,2,4 \
  --runs 5 \
  --artifacts-dir benchmarks/out
```

Outputs:
- `benchmarks/out/benchmark.json`
- `benchmarks/out/benchmark.csv`

Comparison benchmark:

```bash
llminfer compare \
  --model facebook/opt-125m \
  --backends eager,compiled,vllm \
  --batch-sizes 1,2,4 \
  --runs 5 \
  --artifacts-dir benchmarks/compare_out
```

Outputs:
- `benchmarks/compare_out/comparison.json`
- `benchmarks/compare_out/comparison.csv`

You can also write explicit paths with `--json-out` and `--csv-out`.

Dedicated eager-vs-vLLM script:

```bash
python examples/benchmark_vs_vllm.py \
  --model facebook/opt-125m \
  --batch-sizes 1,2,4,8 \
  --runs 5 \
  --continuous \
  --outdir benchmarks/vllm_compare
```

Additional plotting outputs:

```bash
llminfer bench --model facebook/opt-125m --plot-suite-dir plots/bench
llminfer compare --model facebook/opt-125m --backends eager,compiled --plot-suite-dir plots/compare
```

Each plot suite writes:
- `<prefix>_dashboard.png` (combined 2x2 view)
- `<prefix>_throughput.png`
- `<prefix>_latency.png`
- `<prefix>_ttft.png` (if TTFT is available)
- `<prefix>_memory.png`

---

## Backend Comparison

Tensor/pipeline parallel knobs can be applied during compare runs, especially
for vLLM:

```bash
llminfer compare \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --backends eager,vllm \
  --tp-size 2 \
  --pp-size 1 \
  --compile-cudagraphs \
  --batch-sizes 1,2,4,8 \
  --runs 5
```

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

Compiled runtime safety:
- `CompiledBackend` marks CUDAGraph step boundaries before each invocation.
- If a compile/runtime-specific failure occurs (e.g. CUDAGraph overwrite errors),
  it can automatically fall back to eager execution (`compile_fallback_to_eager=True`).

---

## CLI

```bash
# Run inference
llminfer run "Tell me about GPUs" --model facebook/opt-125m

# Streaming
llminfer stream "Explain backpropagation" --model facebook/opt-1.3b --quant nf4

# Benchmark
llminfer bench --model facebook/opt-125m --batch-sizes 1,2,4,8 --runs 10 --plot bench.png

# Benchmark + continuous batching + paged KV
llminfer bench \
  --model facebook/opt-125m \
  --batch-sizes 1,2,4,8 \
  --runs 10 \
  --continuous \
  --paged-kv \
  --page-size-tokens 16 \
  --plot-suite-dir plots/bench

# Compare backends
llminfer compare --model facebook/opt-125m --backends eager,compiled,vllm --plot compare.png

# Compare backends + full plot suite
llminfer compare --model facebook/opt-125m --backends eager,vllm --continuous --plot-suite-dir plots/compare

# Run OpenAI-compatible server
llminfer serve --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000

# Info
llminfer info --model facebook/opt-125m --backend compiled
```

---

## Serving API (OpenAI-Compatible)

`llminfer serve` launches a FastAPI server with an internal continuous-batching
scheduler (`max_batch_size`, `batch_timeout_ms`, `max_queue_size`).

Endpoints:
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /metrics` (Prometheus format, if `prometheus-client` installed)

Start server:

```bash
llminfer serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --backend eager \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch-size 16 \
  --batch-timeout-ms 20 \
  --max-queue-size 1024
```

Non-streaming chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llminfer",
    "messages": [
      {"role":"system","content":"You are concise."},
      {"role":"user","content":"Tell me about GPUs in 3 bullets."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

Non-streaming semantics:
- one JSON response (`object: "chat.completion"`)
- final model output is in `choices[0].message.content`
- token counts are in `usage`

Python example (non-streaming):

```python
import requests

payload = {
    "model": "local-llminfer",
    "messages": [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Tell me about GPUs in 4 bullet points."},
    ],
    "max_tokens": 160,
    "temperature": 0.2,
}

resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=180)
resp.raise_for_status()
data = resp.json()
print(data["choices"][0]["message"]["content"])
print(data["usage"])
```

Streaming chat request (SSE):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"local-llminfer",
    "messages":[{"role":"user","content":"Explain KV cache in under 80 words."}],
    "stream": true
  }'
```

Streaming semantics (SSE):
- chunks arrive as Server-Sent Event lines: `data: {...}`
- text increments are in `choices[0].delta.content`
- stream terminates with `data: [DONE]`

Python example (SSE, robust parser):

```python
import json
import requests

stream_payload = {
    "model": "local-llminfer",
    "messages": [{"role": "user", "content": "Explain KV cache in under 80 words."}],
    "max_tokens": 120,
    "temperature": 0.2,
    "stream": True,
}

decoder = json.JSONDecoder()
done = False

with requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json=stream_payload,
    stream=True,
    timeout=180,
) as r:
    r.raise_for_status()

    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if "data:" not in line:
            continue

        # Some transports can coalesce multiple data frames in one line.
        payloads = [p.strip() for p in line.split("data:") if p.strip()]
        for payload in payloads:
            if payload == "[DONE]":
                done = True
                break

            # Ignore heartbeat/newline fragments.
            if payload in {"\\n", "\\n\\n", "\"\\n\"", "\"\\n\\n\""}:
                continue

            s = payload
            while s:
                s = s.lstrip()
                if not s:
                    break
                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    break
                s = s[idx:]
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)

        if done:
            print("\\n[done]")
            break
```

This parser pattern is also used in:
- `examples/llminfer_serving_colab.ipynb`
- `examples/openai_api_client.py`

Health and metrics:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/metrics
```

---

## Project Structure

```
llminfer/
├── __init__.py          Public API surface
├── config.py            EngineConfig, QuantConfig, CacheConfig
├── engine.py            InferenceEngine (main entrypoint)
├── request.py           GenerationRequest, GenerationResult, StreamChunk
├── kv_cache.py          KVCacheManager with prefix cache + optional paged KV layout
├── batching.py          Dynamic batch assembly (sync + async)
├── streaming.py         TokenStreamer wrapping HF TextIteratorStreamer
├── benchmark.py         Benchmarker + BackendComparison + artifact export
├── serving.py           ContinuousBatchScheduler + chat prompt helpers
├── api.py               OpenAI-compatible FastAPI app + SSE streaming
├── cli.py               Typer CLI (run / stream / bench / compare / serve / info)
└── backends/
    ├── base.py          Abstract BaseBackend
    ├── eager.py         HF generation + quantization + constraints + adapters
    ├── compiled.py      torch.compile backend + runtime fallback to eager
    └── vllm_backend.py  vLLM async backend + true token streaming
```

examples/
├── full_demo.py                     Original end-to-end walkthrough
├── advanced_features.py             Constraints, speculative setup, artifacts, adapters
├── benchmark_vs_vllm.py             Focused eager-vs-vLLM benchmark/export script
├── run_server.py                    Launch OpenAI-compatible server
├── openai_api_client.py             Non-stream + stream API client examples
├── llminfer_all_examples_colab.ipynb Master Colab that covers all example paths
├── llminfer_colab.ipynb             Quickstart notebook
├── llminfer_readme_colab.ipynb      Concise README companion notebook
├── llminfer_advanced_colab.ipynb    Advanced tuning notebook
└── llminfer_serving_colab.ipynb     Serving/API notebook

## Examples and Colab Notebooks

### Python scripts

- `examples/full_demo.py`
  - Original end-to-end package tour (eager, quantization, streaming, batch,
    prefix cache, benchmark, backend comparison).

- `examples/advanced_features.py`
  - Demonstrates advanced decoding controls.
  - Shows optional speculative-decoding setup with `assistant_model_name`.
  - Shows optional PEFT adapter workflow.
  - Exports benchmark artifacts (`benchmark.json`, `benchmark.csv`, `benchmark.png`).

- `examples/benchmark_vs_vllm.py`
  - Compares eager and vLLM backends with the same model/prompts.
  - Supports HF loading flags, continuous batching, and paged-KV toggles.
  - Exports JSON/CSV plus dashboard and per-metric plots.

- `examples/run_server.py`
  - Starts OpenAI-compatible API with `ContinuousBatchScheduler`.
  - Intended as a minimal serving bootstrap script.

- `examples/openai_api_client.py`
  - Calls `/v1/chat/completions` in non-streaming mode.
  - Parses SSE streaming chunks for streaming mode.

### Colab notebooks

The notebooks are intentionally de-duplicated so each has a distinct job:

- `examples/llminfer_all_examples_colab.ipynb`
  - End-to-end master notebook covering all example scripts/workflows.
  - Includes clear section headers, explicit prints, artifact exports, and plot displays.

- `examples/llminfer_colab.ipynb`
  - Minimal quickstart: install, run, stream, static vs continuous batching, benchmark plots.

- `examples/llminfer_readme_colab.ipynb`
  - Concise runnable companion to README sections.
  - Links out to specialized notebooks instead of duplicating all steps.

- `examples/llminfer_advanced_colab.ipynb`
  - Advanced controls: HF loading flags, speculative knobs, compile/cudagraph toggles,
    continuous batching, artifact suite, optional vLLM compare.
  - Includes vLLM install/import checks and guidance for torch/vLLM ABI mismatch in Colab.

- `examples/llminfer_serving_colab.ipynb`
  - OpenAI-compatible server workflow with robust startup checks.
  - Non-streaming + robust SSE streaming parser, concurrency smoke test, metrics checks.

Notebook quality conventions:
- clean cells (no stale output blobs committed),
- explicit print summaries for each section,
- saved plots rendered inline from generated artifacts,
- optional heavy sections (`RUN_VLLM`, etc.) gated with clear flags.

vLLM note for Colab:
- If you see an import error with `undefined symbol ... _C.abi3.so`, your runtime
  has a torch/vLLM binary mismatch. Restart the runtime, rerun install cells, and
  use the notebook's vLLM section (it force-refreshes vLLM and validates import).

---

## Key Design Decisions

**Why KV cache at the application level?**  
HuggingFace manages KV cache internally per `generate()` call. Our
`KVCacheManager` adds *prefix-level* caching across calls — identical
system prompts don't re-compute their KV blocks on the second request.
It can also represent cached KV in fixed-size pages to reduce memory
fragmentation pressure in long-running processes.

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
- `fastapi` + `uvicorn` + `prometheus-client` for `llminfer serve`
- `peft` for LoRA adapter hot-swap

---

## License

This project is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later). See the [LICENSE](LICENSE) file.
