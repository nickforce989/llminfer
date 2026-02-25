"""
examples/advanced_features.py

Showcases newly added capabilities:
  1) Constrained decoding controls
  2) Speculative decoding (assistant model)
  3) Benchmark artifact export (JSON + CSV)
  4) Optional LoRA adapter hot-swap
"""

from __future__ import annotations

from pathlib import Path

from llminfer import Benchmarker, EngineConfig, InferenceEngine
from llminfer.config import Backend

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ASSISTANT_MODEL = None  # e.g. "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    cfg = EngineConfig(
        model_name=MODEL,
        backend=Backend.EAGER,
        assistant_model_name=ASSISTANT_MODEL,
        max_batch_size=8,
        batch_timeout_ms=20,
    )
    engine = InferenceEngine(cfg)
    engine.load()

    print("\n=== Constrained decoding ===")
    constrained = engine.run(
        "Write two concise bullet points about GPUs for deep learning.",
        max_new_tokens=80,
        temperature=0.2,
        no_repeat_ngram_size=3,
        bad_words=["joking", "not sure"],
        stop_sequences=["\n\n"],
        seed=7,
    )
    print(constrained.generated_text)

    print("\n=== Optional adapter hot-swap (PEFT) ===")
    adapter_path = None  # set to local path or HF repo id
    if adapter_path:
        try:
            engine.load_adapter(adapter_path, adapter_name="demo")
            print("Loaded adapters:", engine.list_adapters())
            adapted = engine.run("Summarize LoRA adapters in one paragraph.", max_new_tokens=96)
            print(adapted.generated_text)
            engine.unload_adapter("demo")
        except Exception as exc:
            print("Adapter demo skipped:", exc)
    else:
        print("Set adapter_path in this script to test adapter loading.")

    print("\n=== Benchmark artifact export ===")
    outdir = Path("benchmarks/artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    bm = Benchmarker(engine)
    result = bm.run(batch_sizes=[1, 2, 4], num_runs=3, max_new_tokens=48)
    result.print_summary()
    result.save_json(str(outdir / "benchmark.json"))
    result.save_csv(str(outdir / "benchmark.csv"))
    result.plot(str(outdir / "benchmark.png"))
    result.plot_suite(str(outdir), prefix="benchmark")
    print(f"Saved artifacts in: {outdir.resolve()}")

    engine.unload()


if __name__ == "__main__":
    main()
