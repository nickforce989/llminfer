"""
examples/run_server.py

Launch the OpenAI-compatible API with continuous batching.
"""

from llminfer import EngineConfig, InferenceEngine
from llminfer.api import create_openai_app
from llminfer.config import Backend, QuantConfig, QuantMode
from llminfer.serving import ContinuousBatchScheduler


def main() -> None:
    import uvicorn

    cfg = EngineConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend=Backend.EAGER,
        quant=QuantConfig(mode=QuantMode.NONE),
        max_batch_size=16,
        batch_timeout_ms=20,
    )

    engine = InferenceEngine(cfg)
    scheduler = ContinuousBatchScheduler(
        engine=engine,
        max_batch_size=cfg.max_batch_size,
        batch_timeout_ms=cfg.batch_timeout_ms,
        max_queue_size=1024,
    )
    app = create_openai_app(engine=engine, scheduler=scheduler)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
