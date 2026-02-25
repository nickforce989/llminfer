"""
examples/openai_api_client.py

Call llminfer's OpenAI-compatible endpoints.
Requires the server to be running at http://127.0.0.1:8000.
"""

from __future__ import annotations

import json

import requests

BASE_URL = "http://127.0.0.1:8000"


def chat_completion() -> None:
    payload = {
        "model": "local-llminfer",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Tell me about GPUs in 3 bullet points."},
        ],
        "max_tokens": 120,
        "temperature": 0.2,
    }
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def streaming_chat_completion() -> None:
    payload = {
        "model": "local-llminfer",
        "messages": [{"role": "user", "content": "Give me a short explanation of KV cache."}],
        "max_tokens": 96,
        "temperature": 0.2,
        "stream": True,
    }
    with requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                chunk = line[6:]
                if chunk == "[DONE]":
                    print("\n[stream done]")
                    break
                obj = json.loads(chunk)
                delta = obj["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)


if __name__ == "__main__":
    chat_completion()
    print("\n--- streaming ---")
    streaming_chat_completion()
