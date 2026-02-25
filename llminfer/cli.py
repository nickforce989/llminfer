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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path or repo id for LoRA adapter"),
    adapter_name: str = typer.Option("default", "--adapter-name"),
) -> None:
    """Run inference on a single prompt."""
    from llminfer import InferenceEngine
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        max_new_tokens=max_tokens,
        assistant_model_name=assistant_model,
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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Path or repo id for LoRA adapter"),
    adapter_name: str = typer.Option("default", "--adapter-name"),
) -> None:
    """Stream token-by-token output."""
    from llminfer import InferenceEngine
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        assistant_model_name=assistant_model,
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
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    bs_list = [int(x) for x in batch_sizes.split(",")]

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        max_batch_size=max(bs_list),
    )
    engine = InferenceEngine(cfg)
    engine.load()

    bm = Benchmarker(engine)
    result = bm.run(batch_sizes=bs_list, num_runs=num_runs, max_new_tokens=max_tokens)
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

    cmp = BackendComparison(model_name=model, backends=backend_list, quant_mode=qm)
    results = cmp.run(batch_sizes=bs_list, num_runs=num_runs)
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
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode
    from llminfer.serving import ContinuousBatchScheduler

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        assistant_model_name=assistant_model,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
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
) -> None:
    """Show engine configuration and model info."""
    from llminfer import InferenceEngine
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode
    from rich.panel import Panel
    from rich.pretty import Pretty

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
    )
    engine = InferenceEngine(cfg)
    engine.load()

    console.print(Panel(Pretty(engine.info()), title="Engine Info", border_style="cyan"))
    console.print(Panel(Pretty(engine.cache_stats()), title="Cache Stats", border_style="yellow"))


if __name__ == "__main__":
    app()
