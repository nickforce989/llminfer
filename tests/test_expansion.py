import asyncio
import json

from llminfer.benchmark import (
    Benchmarker,
    BenchmarkResult,
    RunMetrics,
    save_comparison_csv,
    save_comparison_json,
    save_comparison_plot_suite,
    save_plot_suite,
)
from llminfer.backends.compiled import CompiledBackend
from llminfer.backends.eager import EagerBackend
from llminfer.config import EngineConfig
from llminfer.request import GenerationRequest, GenerationResult, TokenStats
from llminfer.serving import ContinuousBatchScheduler, chat_messages_to_prompt


def test_chat_messages_to_prompt():
    prompt = chat_messages_to_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain GPUs."},
        ]
    )
    assert "SYSTEM: You are concise." in prompt
    assert "USER: Explain GPUs." in prompt
    assert prompt.endswith("ASSISTANT:")


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def run_requests(self, requests):
        self.calls.append([r.request_id for r in requests])
        out = []
        for req in requests:
            out.append(
                GenerationResult(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    generated_text=f"ok:{req.prompt}",
                    stats=TokenStats(prompt_tokens=1, generated_tokens=2, total_latency_ms=1),
                )
            )
        return out


def test_continuous_batch_scheduler_groups_requests():
    async def _run():
        engine = _FakeEngine()
        scheduler = ContinuousBatchScheduler(
            engine=engine,
            max_batch_size=2,
            batch_timeout_ms=200,
            max_queue_size=16,
        )
        await scheduler.start()

        reqs = [GenerationRequest(prompt=f"p{i}") for i in range(3)]
        results = await asyncio.gather(*(scheduler.submit(r) for r in reqs))

        await scheduler.stop()
        return engine, results

    engine, results = asyncio.run(_run())

    assert len(results) == 3
    assert [r.generated_text for r in results] == ["ok:p0", "ok:p1", "ok:p2"]
    # With max_batch_size=2 and three requests, we expect two backend calls.
    assert len(engine.calls) == 2
    assert len(engine.calls[0]) == 2
    assert len(engine.calls[1]) == 1


def test_benchmark_artifact_exports(tmp_path):
    result = BenchmarkResult(backend_name="eager", model_name="test-model", quant_mode="none")
    m = RunMetrics(batch_size=2, prompt_type="short")
    m.latencies_ms = [10.0, 12.0, 11.0]
    m.throughputs_tps = [100.0, 110.0, 90.0]
    m.ttfts_ms = [3.0, 4.0, 3.5]
    m.gpu_memory_mb = 123.0
    result.metrics_by_batch[2] = m

    json_path = tmp_path / "bench.json"
    csv_path = tmp_path / "bench.csv"
    result.save_json(str(json_path))
    result.save_csv(str(csv_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["result"]["backend"] == "eager"
    assert csv_path.read_text(encoding="utf-8").startswith("backend,model,quant_mode")


def test_comparison_artifact_exports(tmp_path):
    result = BenchmarkResult(backend_name="compiled", model_name="m", quant_mode="none")
    m = RunMetrics(batch_size=1, prompt_type="short")
    m.latencies_ms = [20.0]
    m.throughputs_tps = [50.0]
    result.metrics_by_batch[1] = m

    json_path = tmp_path / "cmp.json"
    csv_path = tmp_path / "cmp.csv"

    save_comparison_json([result], str(json_path))
    save_comparison_csv([result], str(csv_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["backend"] == "compiled"
    assert "batch_size" in csv_path.read_text(encoding="utf-8")


def test_plot_suite_exports(tmp_path):
    result = BenchmarkResult(backend_name="eager", model_name="m", quant_mode="none")
    m = RunMetrics(batch_size=1, prompt_type="short")
    m.latencies_ms = [20.0, 21.0]
    m.throughputs_tps = [50.0, 49.0]
    m.ttfts_ms = [4.0, 5.0]
    m.gpu_memory_mb = 111.0
    result.metrics_by_batch[1] = m

    plots = save_plot_suite([result], output_dir=str(tmp_path), prefix="bench")
    assert isinstance(plots, dict)
    # If matplotlib is available, at least dashboard should exist.
    if plots:
        assert "dashboard" in plots


def test_comparison_plot_suite_exports(tmp_path):
    result = BenchmarkResult(backend_name="compiled", model_name="m", quant_mode="none")
    m = RunMetrics(batch_size=2, prompt_type="short")
    m.latencies_ms = [30.0]
    m.throughputs_tps = [75.0]
    m.gpu_memory_mb = 222.0
    result.metrics_by_batch[2] = m

    plots = save_comparison_plot_suite([result], output_dir=str(tmp_path), prefix="cmp")
    assert isinstance(plots, dict)
    if plots:
        assert "throughput" in plots


def test_compiled_error_detection_helper():
    err = RuntimeError("Error: accessing tensor output of CUDAGraphs that has been overwritten")
    assert CompiledBackend._looks_like_compile_runtime_error(err)
    assert not CompiledBackend._looks_like_compile_runtime_error(RuntimeError("plain error"))


def test_eager_extract_unused_model_kwargs():
    keys = EagerBackend._extract_unused_model_kwargs(
        ValueError("The following `model_kwargs` are not used by the model: ['a', 'b']")
    )
    assert keys == ["a", "b"]
    assert EagerBackend._extract_unused_model_kwargs(ValueError("other")) == []


def test_eager_generate_retries_dropping_unsupported_kwargs():
    backend = EagerBackend(EngineConfig())

    class _DummyModel:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError(
                    "The following `model_kwargs` are not used by the model: ['num_assistant_tokens']"
                )
            return "ok"

    backend._model = _DummyModel()
    out = backend._generate_with_supported_kwargs(
        inputs={"input_ids": "x"},
        gen_kwargs={"num_assistant_tokens": 8, "temperature": 0.2},
    )
    assert out == "ok"
    assert backend._model.calls == 2


def test_benchmarker_can_use_continuous_batching():
    class _BenchEngine:
        def __init__(self):
            self.cfg = type(
                "Cfg",
                (),
                {
                    "backend": type("B", (), {"value": "eager"})(),
                    "model_name": "dummy-model",
                    "quant": type("Q", (), {"mode": type("M", (), {"value": "none"})()})(),
                },
            )()
            self.called_run_batch = 0
            self.called_run_batch_cont = 0

        def run_batch(self, prompts, max_new_tokens=128):
            self.called_run_batch += 1
            return [
                GenerationResult(
                    request_id=f"id-{i}",
                    prompt=p,
                    generated_text="ok",
                    stats=TokenStats(generated_tokens=4, total_latency_ms=1),
                )
                for i, p in enumerate(prompts)
            ]

        def run_batch_continuous(self, prompts, max_new_tokens=128):
            self.called_run_batch_cont += 1
            return self.run_batch(prompts, max_new_tokens=max_new_tokens)

    engine = _BenchEngine()
    bm = Benchmarker(engine)
    bm.run(
        batch_sizes=[1],
        num_runs=1,
        warmup_runs=1,
        max_new_tokens=8,
        use_continuous_batching=True,
    )
    assert engine.called_run_batch_cont > 0
