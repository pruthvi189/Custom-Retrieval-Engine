"""Provider clients - OpenRouter for embeddings and generation.

Keys come from env vars only, never from source files.
"""

from __future__ import annotations

import os

import httpx

EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL") or "openai/text-embedding-3-small"
GEN_MODEL = os.environ.get("OPENROUTER_GEN_MODEL") or "meta-llama/llama-3.3-70b-instruct"
_embed_key = os.environ.get("OPENROUTER_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")


def embed_one(text: str) -> list[float] | None:
    if not _embed_key:
        return None
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": "Bearer " + _embed_key,
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": text},
            timeout=30.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    d = r.json()
    e = None
    if isinstance(d, dict):
        data = d.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            e = data[0].get("embedding")
    return e if isinstance(e, list) and e else None


def embed_available() -> bool:
    if not _embed_key:
        return False
    try:
        e = embed_one("test")
        return bool(e)
    except Exception:
        return False


def generate(prompt: str) -> str:
    """Generate using OpenRouter (OpenAI-compatible)."""
    if not _embed_key:
        return "ERROR: OPENROUTER_API_KEY is not set."
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + _embed_key,
                "Content-Type": "application/json",
            },
            json={
                "model": GEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False,
            },
            timeout=60.0,
        )
        if r.status_code != 200:
            msg = f"OpenRouter API request failed (HTTP {r.status_code})"
            try:
                d = r.json()
                if isinstance(d, dict) and isinstance(d.get("error"), dict) and d["error"].get("message"):
                    msg = d["error"]["message"]
            except Exception:
                pass
            return "ERROR: " + msg
        d = r.json()
        ans = None
        if isinstance(d, dict):
            choices = d.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    ans = message.get("content")
        return ans if ans else "ERROR: Empty response from OpenRouter"
    except httpx.HTTPError:
        return "ERROR: Could not reach the OpenRouter API. Check your internet connection."


def openrouter_available() -> bool:
    if not _embed_key:
        return False
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": "Bearer " + _embed_key},
            timeout=15.0,
        )
        return r.status_code == 200
    except Exception:
        return False


# Backward compatibility
def groq_available() -> bool:
    return openrouter_available()
