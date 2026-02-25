"""
OpenAI-compatible HTTP API for llminfer.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from llminfer.engine import InferenceEngine
from llminfer.request import GenerationRequest
from llminfer.serving import ContinuousBatchScheduler, chat_messages_to_prompt

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - optional dependency
    CONTENT_TYPE_LATEST = "text/plain"
    Counter = Gauge = Histogram = None
    generate_latest = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = 0.2
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    no_repeat_ngram_size: int = 0
    bad_words: Optional[List[str]] = None
    force_words: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = 0.2
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    no_repeat_ngram_size: int = 0
    bad_words: Optional[List[str]] = None
    force_words: Optional[List[str]] = None


class Metrics:
    def __init__(self) -> None:
        if Counter is None:
            self.enabled = False
            return
        self.enabled = True
        self.requests = Counter(
            "llminfer_requests_total",
            "Total API requests",
            ["endpoint", "stream"],
        )
        self.errors = Counter(
            "llminfer_errors_total",
            "Total API errors",
            ["endpoint"],
        )
        self.latency = Histogram(
            "llminfer_request_latency_seconds",
            "API latency in seconds",
            ["endpoint", "stream"],
        )
        self.inflight = Gauge(
            "llminfer_inflight_requests",
            "In-flight requests",
            ["endpoint"],
        )
        self.queue_size = Gauge(
            "llminfer_scheduler_queue_size",
            "Scheduler queue depth",
        )


class _Measure:
    def __init__(self, metrics: Metrics, endpoint: str, stream: bool):
        self.metrics = metrics
        self.endpoint = endpoint
        self.stream = "1" if stream else "0"
        self._start = 0.0

    def __enter__(self):
        if self.metrics.enabled:
            self.metrics.requests.labels(self.endpoint, self.stream).inc()
            self.metrics.inflight.labels(self.endpoint).inc()
            self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, _tb):
        if not self.metrics.enabled:
            return False
        if exc_type is not None:
            self.metrics.errors.labels(self.endpoint).inc()
        elapsed = max(time.monotonic() - self._start, 0.0)
        self.metrics.latency.labels(self.endpoint, self.stream).observe(elapsed)
        self.metrics.inflight.labels(self.endpoint).dec()
        return False


def _normalize_stop(stop_value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if stop_value is None:
        return None
    if isinstance(stop_value, str):
        return [stop_value]
    return [s for s in stop_value if s]


def _usage_from_stats(stats) -> Dict[str, int]:
    prompt_tokens = int(getattr(stats, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(stats, "generated_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _build_request(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    seed: Optional[int],
    no_repeat_ngram_size: int,
    bad_words: Optional[List[str]],
    force_words: Optional[List[str]],
) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop,
        seed=seed,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words=bad_words,
        force_words=force_words,
    )


def _sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\\n\\n"


def create_openai_app(
    engine: InferenceEngine,
    scheduler: Optional[ContinuousBatchScheduler] = None,
    model_alias: Optional[str] = None,
):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "FastAPI is required for serving. Install with `pip install fastapi uvicorn`"
        ) from exc

    app = FastAPI(title="llminfer API", version="0.2.0")
    metrics = Metrics()

    nonlocal_scheduler = scheduler or ContinuousBatchScheduler(
        engine,
        max_batch_size=engine.cfg.max_batch_size,
        batch_timeout_ms=engine.cfg.batch_timeout_ms,
    )
    served_model_name = model_alias or engine.cfg.model_name

    @app.on_event("startup")
    async def _startup() -> None:
        await nonlocal_scheduler.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await nonlocal_scheduler.stop()
        await asyncio.to_thread(engine.unload)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        if metrics.enabled:
            metrics.queue_size.set(nonlocal_scheduler.queue_size)
        return {
            "status": "ok",
            "model": served_model_name,
            "backend": engine.cfg.backend.value,
            "queue_size": nonlocal_scheduler.queue_size,
        }

    @app.get("/metrics")
    async def metrics_endpoint():
        if not metrics.enabled or generate_latest is None:
            raise HTTPException(status_code=404, detail="prometheus_client not installed")
        metrics.queue_size.set(nonlocal_scheduler.queue_size)
        data = generate_latest()
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "llminfer",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        stop = _normalize_stop(req.stop)

        if req.stream:
            with _Measure(metrics, endpoint="completions", stream=True):
                return StreamingResponse(
                    _stream_completion(engine, served_model_name, prompt, req, stop),
                    media_type="text/event-stream",
                )

        with _Measure(metrics, endpoint="completions", stream=False):
            gen_request = _build_request(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
                seed=req.seed,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                bad_words=req.bad_words,
                force_words=req.force_words,
            )
            result = await nonlocal_scheduler.submit(gen_request)

        now = int(time.time())
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": now,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "text": result.generated_text,
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage_from_stats(result.stats),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        prompt = chat_messages_to_prompt([m.model_dump() for m in req.messages])
        stop = _normalize_stop(req.stop)

        if req.stream:
            with _Measure(metrics, endpoint="chat_completions", stream=True):
                return StreamingResponse(
                    _stream_chat_completion(engine, served_model_name, prompt, req, stop),
                    media_type="text/event-stream",
                )

        with _Measure(metrics, endpoint="chat_completions", stream=False):
            gen_request = _build_request(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
                seed=req.seed,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                bad_words=req.bad_words,
                force_words=req.force_words,
            )
            result = await nonlocal_scheduler.submit(gen_request)

        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": now,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.generated_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage_from_stats(result.stats),
        }

    return app


async def _stream_completion(
    engine: InferenceEngine,
    model_name: str,
    prompt: str,
    req: CompletionRequest,
    stop: Optional[List[str]],
):
    out_q: queue.Queue = queue.Queue()

    def _run() -> None:
        try:
            for chunk in engine.stream(
                prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop_sequences=stop,
                seed=req.seed,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                bad_words=req.bad_words,
                force_words=req.force_words,
            ):
                out_q.put(chunk)
        except Exception as exc:  # pragma: no cover - passthrough
            out_q.put(exc)
        finally:
            out_q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    created = int(time.time())
    chunk_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    while True:
        item = await asyncio.to_thread(out_q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        if not item.is_final:
            payload = {
                "id": chunk_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": item.token,
                        "finish_reason": None,
                    }
                ],
            }
            yield _sse_data(payload)
            continue

        payload = {
            "id": chunk_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage_from_stats(item.stats),
        }
        yield _sse_data(payload)
        yield "data: [DONE]\\n\\n"
        break


async def _stream_chat_completion(
    engine: InferenceEngine,
    model_name: str,
    prompt: str,
    req: ChatCompletionRequest,
    stop: Optional[List[str]],
):
    out_q: queue.Queue = queue.Queue()

    def _run() -> None:
        try:
            for chunk in engine.stream(
                prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop_sequences=stop,
                seed=req.seed,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                bad_words=req.bad_words,
                force_words=req.force_words,
            ):
                out_q.put(chunk)
        except Exception as exc:  # pragma: no cover - passthrough
            out_q.put(exc)
        finally:
            out_q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    created = int(time.time())
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # Initial role delta
    first_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield _sse_data(first_payload)

    while True:
        item = await asyncio.to_thread(out_q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        if not item.is_final:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": item.token}, "finish_reason": None}],
            }
            yield _sse_data(payload)
            continue

        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": _usage_from_stats(item.stats),
        }
        yield _sse_data(payload)
        yield "data: [DONE]\\n\\n"
        break
