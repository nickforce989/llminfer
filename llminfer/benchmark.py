"""
Benchmarker — throughput, latency, and backend comparison.

Produces:
  - Tokens/sec vs batch size curve
  - Latency (ms) vs batch size curve
  - Time-to-first-token (TTFT) distribution
  - GPU memory usage per configuration
  - Side-by-side table: eager vs compiled vs vllm

Usage
-----
from llminfer import InferenceEngine, EngineConfig, Benchmarker
from llminfer.config import Backend

engine = InferenceEngine(EngineConfig(model_name="facebook/opt-125m"))
engine.load()

bm = Benchmarker(engine)
result = bm.run(batch_sizes=[1, 2, 4, 8], num_runs=10)
result.print_summary()
result.plot("bench_results.png")
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Synthetic prompts of varying lengths for benchmarking
_PROMPTS = {
    "short":  "What is the capital of France?",
    "medium": (
        "Explain the key differences between supervised and unsupervised "
        "machine learning, and provide two real-world examples of each."
    ),
    "long": (
        "You are an expert ML systems engineer. Describe in detail the "
        "architecture of a production-grade LLM serving system, including "
        "the role of the scheduler, memory manager, attention kernel, "
        "tokenization pipeline, and how continuous batching differs from "
        "static batching. Also discuss trade-offs between latency and "
        "throughput optimizations, and how KV cache eviction policies "
        "impact serving quality under heavy load."
    ),
}


@dataclass
class RunMetrics:
    batch_size: int
    prompt_type: str
    latencies_ms: List[float] = field(default_factory=list)
    throughputs_tps: List[float] = field(default_factory=list)   # tokens/sec
    ttfts_ms: List[float] = field(default_factory=list)
    gpu_memory_mb: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        s = sorted(self.latencies_ms)
        return s[len(s) // 2] if s else 0.0

    @property
    def p95_latency_ms(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)] if s else 0.0

    @property
    def p99_latency_ms(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.99)] if s else 0.0

    @property
    def mean_throughput_tps(self) -> float:
        return statistics.mean(self.throughputs_tps) if self.throughputs_tps else 0.0

    @property
    def mean_ttft_ms(self) -> float:
        return statistics.mean(self.ttfts_ms) if self.ttfts_ms else 0.0


@dataclass
class BenchmarkResult:
    backend_name: str
    model_name: str
    quant_mode: str
    metrics_by_batch: Dict[int, RunMetrics] = field(default_factory=dict)

    def print_summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            _rich_summary(self)
        except ImportError:
            _plain_summary(self)

    def plot(self, output_path: str = "benchmark.png") -> None:
        _plot_results([self], output_path)

    @property
    def batch_sizes(self) -> List[int]:
        return sorted(self.metrics_by_batch.keys())

    @property
    def throughput_curve(self) -> Tuple[List[int], List[float]]:
        xs = self.batch_sizes
        ys = [self.metrics_by_batch[b].mean_throughput_tps for b in xs]
        return xs, ys

    @property
    def latency_curve(self) -> Tuple[List[int], List[float]]:
        xs = self.batch_sizes
        ys = [self.metrics_by_batch[b].mean_latency_ms for b in xs]
        return xs, ys


class Benchmarker:
    """
    Runs structured benchmarks on an InferenceEngine instance.

    Parameters
    ----------
    engine : InferenceEngine
        A loaded engine to benchmark.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    def run(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_runs: int = 10,
        warmup_runs: int = 3,
        prompt_type: str = "medium",
        max_new_tokens: int = 128,
    ) -> BenchmarkResult:
        """
        Run latency and throughput benchmarks.

        Parameters
        ----------
        batch_sizes   : List of batch sizes to test.
        num_runs      : Number of timed runs per batch size.
        warmup_runs   : Untimed warmup runs before measurement.
        prompt_type   : One of 'short', 'medium', 'long'.
        max_new_tokens: Token budget per request.
        """
        cfg = self.engine.cfg
        result = BenchmarkResult(
            backend_name=cfg.backend.value,
            model_name=cfg.model_name,
            quant_mode=cfg.quant.mode.value,
        )

        prompt = _PROMPTS.get(prompt_type, _PROMPTS["medium"])
        logger.info(
            "Starting benchmark  backend=%s  batch_sizes=%s  runs=%d",
            cfg.backend.value,
            batch_sizes,
            num_runs,
        )

        for bs in batch_sizes:
            logger.info("  Benchmarking batch_size=%d ...", bs)
            prompts = [prompt] * bs

            # Warmup
            for _ in range(warmup_runs):
                self.engine.run_batch(prompts, max_new_tokens=max_new_tokens)
            _flush_cuda_cache()

            # Timed runs
            metrics = RunMetrics(batch_size=bs, prompt_type=prompt_type)
            for run_idx in range(num_runs):
                t0 = time.monotonic()
                results = self.engine.run_batch(prompts, max_new_tokens=max_new_tokens)
                t1 = time.monotonic()

                latency_ms = (t1 - t0) * 1000
                total_tokens = sum(r.stats.generated_tokens for r in results)
                throughput = total_tokens / max((t1 - t0), 1e-9)

                metrics.latencies_ms.append(latency_ms)
                metrics.throughputs_tps.append(throughput)

                # TTFT from first result (approximate for batch)
                if results[0].stats.time_to_first_token_ms > 0:
                    metrics.ttfts_ms.append(results[0].stats.time_to_first_token_ms)

            metrics.gpu_memory_mb = _get_gpu_memory_mb()
            result.metrics_by_batch[bs] = metrics

            logger.info(
                "    bs=%d  latency=%.1fms  tps=%.1f  mem=%.0fMB",
                bs,
                metrics.mean_latency_ms,
                metrics.mean_throughput_tps,
                metrics.gpu_memory_mb,
            )

        return result


class BackendComparison:
    """
    Compare eager vs compiled vs vllm on the same model.

    Usage
    -----
    from llminfer.benchmark import BackendComparison
    from llminfer.config import Backend

    cmp = BackendComparison(
        model_name="facebook/opt-125m",
        backends=[Backend.EAGER, Backend.COMPILED],
    )
    results = cmp.run(batch_sizes=[1, 4, 8])
    cmp.print_table(results)
    cmp.plot(results, "comparison.png")
    """

    def __init__(
        self,
        model_name: str,
        backends: Optional[List] = None,
        quant_mode=None,
    ) -> None:
        self.model_name = model_name
        self.backends = backends
        self.quant_mode = quant_mode

    def run(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_runs: int = 10,
        max_new_tokens: int = 128,
    ) -> List[BenchmarkResult]:
        from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode
        from llminfer.engine import InferenceEngine

        backends = self.backends or [Backend.EAGER, Backend.COMPILED]
        results: List[BenchmarkResult] = []

        for backend in backends:
            logger.info("=== Benchmarking backend: %s ===", backend.value)
            quant = QuantConfig(mode=self.quant_mode) if self.quant_mode else QuantConfig()
            cfg = EngineConfig(
                model_name=self.model_name,
                backend=backend,
                quant=quant,
            )
            engine = InferenceEngine(cfg)
            engine.load()

            bm = Benchmarker(engine)
            result = bm.run(
                batch_sizes=batch_sizes,
                num_runs=num_runs,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)
            engine.unload()
            _flush_cuda_cache()

        return results

    def print_table(self, results: List[BenchmarkResult]) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            _rich_comparison_table(results)
        except ImportError:
            _plain_comparison_table(results)

    def plot(self, results: List[BenchmarkResult], output_path: str = "comparison.png") -> None:
        _plot_results(results, output_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(results: List[BenchmarkResult], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot.")
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("LLM Inference Benchmark", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_tps   = fig.add_subplot(gs[0, 0])
    ax_lat   = fig.add_subplot(gs[0, 1])
    ax_ttft  = fig.add_subplot(gs[1, 0])
    ax_mem   = fig.add_subplot(gs[1, 1])

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for i, res in enumerate(results):
        label = f"{res.backend_name} ({res.quant_mode})"
        color = colors[i % len(colors)]
        bs_list = res.batch_sizes

        xs, tps_ys = res.throughput_curve
        ax_tps.plot(xs, tps_ys, "o-", color=color, label=label, linewidth=2, markersize=6)

        xs, lat_ys = res.latency_curve
        ax_lat.plot(xs, lat_ys, "s-", color=color, label=label, linewidth=2, markersize=6)

        ttft_ys = [res.metrics_by_batch[b].mean_ttft_ms for b in bs_list]
        if any(t > 0 for t in ttft_ys):
            ax_ttft.plot(bs_list, ttft_ys, "^-", color=color, label=label, linewidth=2, markersize=6)

        mem_ys = [res.metrics_by_batch[b].gpu_memory_mb for b in bs_list]
        ax_mem.bar(
            [b + i * 0.3 for b in range(len(bs_list))],
            mem_ys,
            width=0.25,
            color=color,
            alpha=0.8,
            label=label,
        )

    # Style
    for ax, title, xlabel, ylabel in [
        (ax_tps,  "Throughput vs Batch Size",       "Batch Size", "Tokens / sec"),
        (ax_lat,  "Latency vs Batch Size",           "Batch Size", "Latency (ms)"),
        (ax_ttft, "Time to First Token vs Batch Size","Batch Size", "TTFT (ms)"),
        (ax_mem,  "GPU Memory vs Batch Size",        "Batch Size", "Memory (MB)"),
    ]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved benchmark plot to %s", output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Rich / plain text tables
# ---------------------------------------------------------------------------

def _rich_summary(result: BenchmarkResult) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=f"Benchmark: {result.model_name} [{result.backend_name} / {result.quant_mode}]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Batch Size", justify="right")
    table.add_column("Latency p50 (ms)", justify="right")
    table.add_column("Latency p95 (ms)", justify="right")
    table.add_column("Throughput (tok/s)", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("GPU Mem (MB)", justify="right")

    for bs in result.batch_sizes:
        m = result.metrics_by_batch[bs]
        table.add_row(
            str(bs),
            f"{m.p50_latency_ms:.1f}",
            f"{m.p95_latency_ms:.1f}",
            f"{m.mean_throughput_tps:.1f}",
            f"{m.mean_ttft_ms:.1f}" if m.ttfts_ms else "N/A",
            f"{m.gpu_memory_mb:.0f}",
        )

    console.print(table)


def _plain_summary(result: BenchmarkResult) -> None:
    print(f"\n{'='*70}")
    print(f"Benchmark: {result.model_name}  [{result.backend_name} / {result.quant_mode}]")
    print(f"{'='*70}")
    print(f"{'BS':>4}  {'P50 lat':>10}  {'P95 lat':>10}  {'Tok/s':>10}  {'TTFT':>10}  {'GPU MB':>10}")
    print("-" * 70)
    for bs in result.batch_sizes:
        m = result.metrics_by_batch[bs]
        ttft = f"{m.mean_ttft_ms:.1f}" if m.ttfts_ms else "N/A"
        print(
            f"{bs:>4}  {m.p50_latency_ms:>10.1f}  {m.p95_latency_ms:>10.1f}  "
            f"{m.mean_throughput_tps:>10.1f}  {ttft:>10}  {m.gpu_memory_mb:>10.0f}"
        )


def _rich_comparison_table(results: List[BenchmarkResult]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    all_bs = sorted(set(bs for r in results for bs in r.batch_sizes))
    table = Table(title="Backend Comparison", header_style="bold magenta")
    table.add_column("Batch Size", justify="right")
    for r in results:
        table.add_column(f"{r.backend_name}\nTok/s", justify="right")
        table.add_column(f"{r.backend_name}\nLat ms", justify="right")

    for bs in all_bs:
        row = [str(bs)]
        for r in results:
            if bs in r.metrics_by_batch:
                m = r.metrics_by_batch[bs]
                row += [f"{m.mean_throughput_tps:.1f}", f"{m.p50_latency_ms:.1f}"]
            else:
                row += ["—", "—"]
        table.add_row(*row)

    console.print(table)


def _plain_comparison_table(results: List[BenchmarkResult]) -> None:
    all_bs = sorted(set(bs for r in results for bs in r.batch_sizes))
    header = f"{'BS':>4}  " + "  ".join(f"{r.backend_name:>14}" for r in results)
    print("\nBackend Comparison (Tok/s)")
    print(header)
    print("-" * len(header))
    for bs in all_bs:
        row = f"{bs:>4}  "
        for r in results:
            if bs in r.metrics_by_batch:
                tps = r.metrics_by_batch[bs].mean_throughput_tps
                row += f"{tps:>14.1f}  "
            else:
                row += f"{'—':>14}  "
        print(row)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flush_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _get_gpu_memory_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0
