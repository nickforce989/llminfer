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
    temperature: float = typer.Option(1.0, "--temp"),
) -> None:
    """Run inference on a single prompt."""
    from llminfer import InferenceEngine
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
        max_new_tokens=max_tokens,
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Loading model...", total=None)
        engine = InferenceEngine(cfg)
        engine.load()

    result = engine.run(prompt, max_new_tokens=max_tokens, temperature=temperature)

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
) -> None:
    """Stream token-by-token output."""
    from llminfer import InferenceEngine
    from llminfer.config import Backend, EngineConfig, QuantConfig, QuantMode

    cfg = EngineConfig(
        model_name=model,
        backend=Backend(backend),
        quant=QuantConfig(mode=QuantMode(quant)),
    )
    engine = InferenceEngine(cfg)
    engine.load()

    console.print(f"\n[bold green]Streaming:[/bold green]")
    for chunk in engine.stream(prompt, max_new_tokens=max_tokens):
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


@app.command()
def compare(
    model: str = typer.Option("facebook/opt-125m", "--model", "-m"),
    backends: str = typer.Option("eager,compiled", "--backends"),
    quant: str = typer.Option("none", "--quant", "-q"),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes"),
    num_runs: int = typer.Option(5, "--runs"),
    plot: Optional[str] = typer.Option(None, "--plot"),
) -> None:
    """Compare multiple backends side by side."""
    from llminfer.benchmark import BackendComparison
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
