"""
Compare llminfer eager backend against vLLM and export artifacts.

Example:
  python examples/benchmark_vs_vllm.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --batch-sizes 1,2,4,8 \
    --runs 5 \
    --outdir benchmarks/vllm_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llminfer.benchmark import (
    BackendComparison,
    save_comparison_csv,
    save_comparison_json,
    save_comparison_plot_suite,
)
from llminfer.config import Backend, QuantMode


def _parse_batch_sizes(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("batch sizes cannot be empty")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llminfer eager vs vLLM.")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--quant", default="none", choices=[m.value for m in QuantMode])
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--continuous", action="store_true", help="Use continuous batching scheduler")
    parser.add_argument("--paged-kv", action="store_true", help="Enable paged KV cache for non-vLLM backends")
    parser.add_argument("--page-size-tokens", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--no-compile-cudagraphs", action="store_true")
    parser.add_argument("--spec-num-assistant-tokens", type=int, default=None)
    parser.add_argument("--spec-confidence-threshold", type=float, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--outdir", default="benchmarks/vllm_compare")
    args = parser.parse_args()

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmp = BackendComparison(
        model_name=args.model,
        backends=[Backend.EAGER, Backend.VLLM],
        quant_mode=QuantMode(args.quant),
        hf_revision=args.revision,
        hf_token=args.hf_token,
        hf_local_files_only=args.local_files_only,
        hf_trust_remote_code=not args.no_trust_remote_code,
        hf_cache_dir=args.cache_dir,
        paged_kv=args.paged_kv,
        page_size_tokens=args.page_size_tokens,
        tensor_parallel_size=max(1, int(args.tp_size)),
        pipeline_parallel_size=max(1, int(args.pp_size)),
        compile_fullgraph=bool(args.compile_fullgraph),
        compile_cudagraphs=not bool(args.no_compile_cudagraphs),
        speculative_num_assistant_tokens=args.spec_num_assistant_tokens,
        speculative_confidence_threshold=args.spec_confidence_threshold,
    )

    results = cmp.run(
        batch_sizes=batch_sizes,
        num_runs=args.runs,
        max_new_tokens=args.max_new_tokens,
        use_continuous_batching=args.continuous,
    )
    cmp.print_table(results)

    save_comparison_json(results, str(outdir / "comparison.json"))
    save_comparison_csv(results, str(outdir / "comparison.csv"))
    cmp.plot(results, str(outdir / "comparison_dashboard.png"))
    save_comparison_plot_suite(results, output_dir=str(outdir), prefix="comparison")

    print(f"Saved artifacts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
