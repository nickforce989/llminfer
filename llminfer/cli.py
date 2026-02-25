"""
llminfer CLI

Usage examples
--------------
# Run inference
llminfer run "Tell me about GPUs" --model facebook/opt-125m

# Stream output
llminfer stream "Explain transformers" --model facebook/opt-125m

# Benchmark
llminfer bench --model facebook/opt-125m --batch-sizes 1,2,4,8

# Compare backends
llminfer compare --model facebook/opt-125m --backends eager,compiled
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="llminfer",
    help="GPU-Efficient LLM Inference Engine",
    add_completion=False,
)
console = Console()

logging.basicConfig(level=logging.WARNING)


def _build_engine_config(
    *,
    model: str,
    backend: str,
    quant: str,
    max_new_tokens: Optional[int] = None,
    assistant_model: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    batch_timeout_ms: Optional[float] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_local_files_only: bool = False,
    hf_trust_remote_code: bool = True,
    hf_cache_dir: Optional[str] = None,
    paged_kv: bool = False,
    page_size_tokens: int = 16,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    compile_fullgraph: bool = False,
    compile_cudagraphs: bool = True,
    speculative_num_assistant_tokens: Optional[int] = None,
    speculative_confidence_threshold: Optional[float] = None,
):
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        assistant_model_name=assistant_model,
        hf_revision=hf_revision,
        hf_token=hf_token,
        hf_local_files_only=hf_local_files_only,
        hf_trust_remote_code=hf_trust_remote_code,
        hf_cache_dir=hf_cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    if max_new_tokens is not None:
        cfg.max_new_tokens = max_new_tokens
    if max_batch_size is not None:
        cfg.max_batch_size = max_batch_size
    if batch_timeout_ms is not None:
        cfg.batch_timeout_ms = batch_timeout_ms
    cfg.cache.enable_paged_kv = paged_kv
    cfg.cache.page_size_tokens = page_size_tokens
    return cfg


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Input prompt"),
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backend: str = typer.Option("eager", "--backend", "-b", help="eager | compiled | vllm"),
    quant: str = typer.Option("none", "--quant", "-q", help="none | int8 | nf4 | fp4"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temp"),
    top_p: float = typer.Option(0.9, "--top-p"),
    top_k: int = typer.Option(50, "--top-k"),
    repetition_penalty: float = typer.Option(1.0, "--repetition-penalty"),
    no_repeat_ngram_size: int = typer.Option(0, "--no-repeat-ngram-size"),
    stop: Optional[str] = typer.Option(None, "--stop", help="Comma-separated stop sequences"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    bad_words: Optional[str] = typer.Option(None, "--bad-words", help="Comma-separated phrases to avoid"),
    force_words: Optional[str] = typer.Option(None, "--force-words", help="Comma-separated phrases to force"),
    assistant_model: Optional[str] = typer.Option(None, "--assistant-model", help="Speculative decoding assistant model"),
    revision: Optional[str] = typer.Option(None, "--revision", help="HF model revision (branch/tag/commit)"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HF token for gated/private models"),
    local_files_only: bool = typer.Option(False, "--local-files-only", help="Load weights from local HF cache only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="HF cache/download directory"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv", help="Enable paged KV representation"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1, help="Tensor parallel size"),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1, help="Pipeline parallel size"),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path or repo id for LoRA adapter"),
    adapter_name: str = typer.Option("default", "--adapter-name"),
) -> None:
    """Run inference on a single prompt."""
    from llminfer import InferenceEngine
    cfg = _build_engine_config(
        model=model,
        backend=backend,
        quant=quant,
        max_new_tokens=max_tokens,
        assistant_model=assistant_model,
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Loading model...", total=None)
        engine = InferenceEngine(cfg)
        engine.load()
        if adapter:
            engine.load_adapter(adapter, adapter_name=adapter_name)

    stop_sequences = [s.strip() for s in stop.split(",")] if stop else None
    bad_words_list = [s.strip() for s in bad_words.split(",")] if bad_words else None
    force_words_list = [s.strip() for s in force_words.split(",")] if force_words else None

    result = engine.run(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        stop_sequences=stop_sequences,
        seed=seed,
        bad_words=bad_words_list,
        force_words=force_words_list,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )

    console.print(f"\n[bold green]Generated:[/bold green]")
    console.print(result.generated_text)
    console.print(
        f"\n[dim]latency={result.stats.total_latency_ms:.0f}ms  "
        f"tokens={result.stats.generated_tokens}  "
        f"tok/s={result.stats.throughput_tokens_per_sec:.1f}[/dim]"
    )


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="Input prompt"),
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backend: str = typer.Option("eager", "--backend", "-b"),
    quant: str = typer.Option("none", "--quant", "-q"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temp"),
    top_p: float = typer.Option(0.9, "--top-p"),
    stop: Optional[str] = typer.Option(None, "--stop", help="Comma-separated stop sequences"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    no_repeat_ngram_size: int = typer.Option(0, "--no-repeat-ngram-size"),
    assistant_model: Optional[str] = typer.Option(None, "--assistant-model"),
    revision: Optional[str] = typer.Option(None, "--revision", help="HF model revision"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token"),
    local_files_only: bool = typer.Option(False, "--local-files-only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path or repo id for LoRA adapter"),
    adapter_name: str = typer.Option("default", "--adapter-name"),
) -> None:
    """Stream token-by-token output."""
    from llminfer import InferenceEngine
    cfg = _build_engine_config(
        model=model,
        backend=backend,
        quant=quant,
        assistant_model=assistant_model,
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    engine = InferenceEngine(cfg)
    engine.load()
    if adapter:
        engine.load_adapter(adapter, adapter_name=adapter_name)
    stop_sequences = [s.strip() for s in stop.split(",")] if stop else None

    console.print(f"\n[bold green]Streaming:[/bold green]")
    for chunk in engine.stream(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
        seed=seed,
        no_repeat_ngram_size=no_repeat_ngram_size,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    ):
        if not chunk.is_final:
            print(chunk.token, end="", flush=True)
        else:
            print()
            if chunk.stats:
                console.print(
                    f"\n[dim]latency={chunk.stats.total_latency_ms:.0f}ms  "
                    f"tok/s={chunk.stats.throughput_tokens_per_sec:.1f}[/dim]"
                )


@app.command()
def bench(
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backend: str = typer.Option("eager", "--backend", "-b"),
    quant: str = typer.Option("none", "--quant", "-q"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes"),
    num_runs: int = typer.Option(10, "--runs"),
    max_tokens: int = typer.Option(128, "--max-tokens"),
    continuous: bool = typer.Option(False, "--continuous/--no-continuous", help="Benchmark with continuous batching"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token"),
    local_files_only: bool = typer.Option(False, "--local-files-only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
    plot: Optional[str] = typer.Option(None, "--plot", help="Save plot to this path"),
    plot_suite_dir: Optional[str] = typer.Option(
        None,
        "--plot-suite-dir",
        help="Save dashboard + per-metric plots in this directory",
    ),
    json_out: Optional[str] = typer.Option(None, "--json-out", help="Save metrics artifact as JSON"),
    csv_out: Optional[str] = typer.Option(None, "--csv-out", help="Save metrics artifact as CSV"),
    artifacts_dir: Optional[str] = typer.Option(None, "--artifacts-dir", help="Write benchmark.json and benchmark.csv here"),
) -> None:
    """Run throughput / latency benchmark."""
    from llminfer import Benchmarker, InferenceEngine

    bs_list = [int(x) for x in batch_sizes.split(",")]

    cfg = _build_engine_config(
        model=model,
        backend=backend,
        quant=quant,
        max_batch_size=max(bs_list),
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    engine = InferenceEngine(cfg)
    engine.load()

    bm = Benchmarker(engine)
    result = bm.run(
        batch_sizes=bs_list,
        num_runs=num_runs,
        max_new_tokens=max_tokens,
        use_continuous_batching=continuous,
    )
    result.print_summary()

    if plot:
        result.plot(plot)
        console.print(f"[green]Plot saved to {plot}[/green]")
    if plot_suite_dir:
        plots = result.plot_suite(output_dir=plot_suite_dir, prefix="benchmark")
        if plots:
            console.print(f"[green]Plot suite saved to {plot_suite_dir}[/green]")
        else:
            console.print("[yellow]No plot suite generated (likely matplotlib unavailable).[/yellow]")
    if artifacts_dir:
        outdir = Path(artifacts_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        result.save_json(str(outdir / "benchmark.json"))
        result.save_csv(str(outdir / "benchmark.csv"))
        console.print(f"[green]Artifacts saved to {outdir}[/green]")
    if json_out:
        result.save_json(json_out)
        console.print(f"[green]JSON saved to {json_out}[/green]")
    if csv_out:
        result.save_csv(csv_out)
        console.print(f"[green]CSV saved to {csv_out}[/green]")


@app.command()
def compare(
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backends: str = typer.Option("eager,compiled", "--backends"),
    quant: str = typer.Option("none", "--quant", "-q"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes"),
    num_runs: int = typer.Option(5, "--runs"),
    continuous: bool = typer.Option(False, "--continuous/--no-continuous", help="Benchmark with continuous batching"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token"),
    local_files_only: bool = typer.Option(False, "--local-files-only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
    plot: Optional[str] = typer.Option(None, "--plot"),
    plot_suite_dir: Optional[str] = typer.Option(
        None,
        "--plot-suite-dir",
        help="Save dashboard + per-metric comparison plots in this directory",
    ),
    json_out: Optional[str] = typer.Option(None, "--json-out"),
    csv_out: Optional[str] = typer.Option(None, "--csv-out"),
    artifacts_dir: Optional[str] = typer.Option(None, "--artifacts-dir"),
) -> None:
    """Compare multiple backends side by side."""
    from llminfer.benchmark import (
        BackendComparison,
        save_comparison_csv,
        save_comparison_json,
        save_comparison_plot_suite,
    )
    from llminfer.config import Backend, QuantMode

    backend_list = [Backend(b.strip()) for b in backends.split(",")]
    bs_list = [int(x) for x in batch_sizes.split(",")]
    qm = QuantMode(quant)

    cmp = BackendComparison(
        model_name=model,
        backends=backend_list,
        quant_mode=qm,
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    results = cmp.run(
        batch_sizes=bs_list,
        num_runs=num_runs,
        use_continuous_batching=continuous,
    )
    cmp.print_table(results)

    if plot:
        cmp.plot(results, plot)
        console.print(f"[green]Plot saved to {plot}[/green]")
    if plot_suite_dir:
        plots = save_comparison_plot_suite(results, output_dir=plot_suite_dir, prefix="comparison")
        if plots:
            console.print(f"[green]Comparison plot suite saved to {plot_suite_dir}[/green]")
        else:
            console.print("[yellow]No comparison plot suite generated (likely matplotlib unavailable).[/yellow]")
    if artifacts_dir:
        outdir = Path(artifacts_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_comparison_json(results, str(outdir / "comparison.json"))
        save_comparison_csv(results, str(outdir / "comparison.csv"))
        console.print(f"[green]Artifacts saved to {outdir}[/green]")
    if json_out:
        save_comparison_json(results, json_out)
        console.print(f"[green]JSON saved to {json_out}[/green]")
    if csv_out:
        save_comparison_csv(results, csv_out)
        console.print(f"[green]CSV saved to {csv_out}[/green]")


@app.command()
def serve(
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m"),
    backend: str = typer.Option("eager", "--backend", "-b"),
    quant: str = typer.Option("none", "--quant", "-q"),
    assistant_model: Optional[str] = typer.Option(None, "--assistant-model"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    model_alias: Optional[str] = typer.Option(None, "--model-alias"),
    max_batch_size: int = typer.Option(16, "--max-batch-size"),
    batch_timeout_ms: float = typer.Option(20.0, "--batch-timeout-ms"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token"),
    local_files_only: bool = typer.Option(False, "--local-files-only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
    max_queue_size: int = typer.Option(1024, "--max-queue-size"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Run an OpenAI-compatible HTTP server with continuous batching."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "uvicorn is required for serve mode. Install with: pip install fastapi uvicorn prometheus-client"
        ) from exc

    from llminfer import InferenceEngine
    from llminfer.api import create_openai_app
    from llminfer.serving import ContinuousBatchScheduler

    cfg = _build_engine_config(
        model=model,
        backend=backend,
        quant=quant,
        assistant_model=assistant_model,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    engine = InferenceEngine(cfg)
    scheduler = ContinuousBatchScheduler(
        engine=engine,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        max_queue_size=max_queue_size,
    )
    api_app = create_openai_app(engine=engine, scheduler=scheduler, model_alias=model_alias)

    console.print(
        f"[green]Serving[/green] model={model_alias or model} backend={backend} "
        f"at http://{host}:{port}"
    )
    uvicorn.run(api_app, host=host, port=port, log_level=log_level)


@app.command()
def info(
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backend: str = typer.Option("eager", "--backend", "-b"),
    quant: str = typer.Option("none", "--quant", "-q"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token"),
    local_files_only: bool = typer.Option(False, "--local-files-only"),
    trust_remote_code: bool = typer.Option(True, "--trust-remote-code/--no-trust-remote-code"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    paged_kv: bool = typer.Option(False, "--paged-kv/--no-paged-kv"),
    page_size_tokens: int = typer.Option(16, "--page-size-tokens", min=1),
    tensor_parallel_size: int = typer.Option(1, "--tp-size", min=1),
    pipeline_parallel_size: int = typer.Option(1, "--pp-size", min=1),
    compile_fullgraph: bool = typer.Option(False, "--compile-fullgraph/--no-compile-fullgraph"),
    compile_cudagraphs: bool = typer.Option(True, "--compile-cudagraphs/--no-compile-cudagraphs"),
    speculative_num_assistant_tokens: Optional[int] = typer.Option(None, "--spec-num-assistant-tokens"),
    speculative_confidence_threshold: Optional[float] = typer.Option(None, "--spec-confidence-threshold"),
) -> None:
    """Show engine configuration and model info."""
    from llminfer import InferenceEngine
    from rich.panel import Panel
    from rich.pretty import Pretty

    cfg = _build_engine_config(
        model=model,
        backend=backend,
        quant=quant,
        hf_revision=revision,
        hf_token=hf_token,
        hf_local_files_only=local_files_only,
        hf_trust_remote_code=trust_remote_code,
        hf_cache_dir=cache_dir,
        paged_kv=paged_kv,
        page_size_tokens=page_size_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        compile_fullgraph=compile_fullgraph,
        compile_cudagraphs=compile_cudagraphs,
        speculative_num_assistant_tokens=speculative_num_assistant_tokens,
        speculative_confidence_threshold=speculative_confidence_threshold,
    )
    engine = InferenceEngine(cfg)
    engine.load()

    console.print(Panel(Pretty(engine.info()), title="Engine Info", border_style="cyan"))
    console.print(Panel(Pretty(engine.cache_stats()), title="Cache Stats", border_style="yellow"))


if __name__ == "__main__":
    app()
