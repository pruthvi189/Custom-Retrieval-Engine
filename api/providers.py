"""Provider clients - Gemini (free tier) preferred, OpenRouter as fallback.

Keys come from env vars only, never from source files.
"""

from __future__ import annotations

import os

import httpx

# OpenRouter defaults (kept for backward compat / status reporting).
EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL") or "openai/text-embedding-3-small"
GEN_MODEL = os.environ.get("OPENROUTER_GEN_MODEL") or "meta-llama/llama-3.3-70b-instruct"
# Gemini (free tier).
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL") or "gemini-embedding-001"
GEMINI_GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL") or "gemini-3.6-flash"
# Groq (OpenAI-compatible) as a final generation fallback.
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_GEN_MODEL = os.environ.get("GROQ_GEN_MODEL") or "openai/gpt-oss-20b"
_gemini_key = os.environ.get("GEMINI_API_KEY", "")
_embed_key = os.environ.get("OPENROUTER_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")


def _extract_embedding(d) -> list[float] | None:
    e = None
    if isinstance(d, dict):
        data = d.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            e = data[0].get("embedding")
    return e if isinstance(e, list) and e else None


def _gemini_embed(text: str) -> list[float] | None:
    if not _gemini_key:
        return None
    try:
        r = httpx.post(
            GEMINI_BASE + "/embeddings",
            headers={
                "Authorization": "Bearer " + _gemini_key,
                "Content-Type": "application/json",
            },
            json={"model": GEMINI_EMBED_MODEL, "input": text, "dimensions": 1536},
            timeout=30.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    return _extract_embedding(r.json())


def _openrouter_embed(text: str) -> list[float] | None:
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
    return _extract_embedding(r.json())


def embed_one(text: str) -> list[float] | None:
    """Embed text. Prefers Gemini (free tier), falls back to OpenRouter."""
    return _gemini_embed(text) or _openrouter_embed(text)


def embed_available() -> bool:
    try:
        return bool(embed_one("test"))
    except Exception:
        return False


def generate(prompt: str) -> str:
    """Generate using Gemini (free tier) first, then OpenRouter, then Groq."""
    if _gemini_key:
        ans = _gemini_generate(prompt)
        if ans and not ans.startswith("ERROR:"):
            return ans
    ans = _openrouter_generate(prompt)
    if ans and not ans.startswith("ERROR:"):
        return ans
    if _groq_key:
        ans = _groq_generate(prompt)
        if ans and not ans.startswith("ERROR:"):
            return ans
    if not _gemini_key and not _embed_key and not _groq_key:
        return "ERROR: No provider configured. Set GEMINI_API_KEY, OPENROUTER_API_KEY or GROQ_API_KEY."
    return ans or "ERROR: No provider available."


def _openrouter_generate(prompt: str) -> str:
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


def _gemini_generate(prompt: str) -> str:
    if not _gemini_key:
        return "ERROR: GEMINI_API_KEY is not set."
    try:
        r = httpx.post(
            GEMINI_BASE + "/chat/completions",
            headers={
                "Authorization": "Bearer " + _gemini_key,
                "Content-Type": "application/json",
            },
            json={
                "model": GEMINI_GEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False,
            },
            timeout=60.0,
        )
    except httpx.HTTPError:
        return "ERROR: Could not reach the Gemini API. Check your internet connection."
    if r.status_code != 200:
        msg = f"Gemini API request failed (HTTP {r.status_code})"
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
    return ans if ans else "ERROR: Empty response from Gemini"


def _groq_generate(prompt: str) -> str:
    if not _groq_key:
        return "ERROR: GROQ_API_KEY is not set."
    try:
        r = httpx.post(
            GROQ_BASE + "/chat/completions",
            headers={
                "Authorization": "Bearer " + _groq_key,
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_GEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False,
            },
            timeout=60.0,
        )
    except httpx.HTTPError:
        return "ERROR: Could not reach the Groq API. Check your internet connection."
    if r.status_code != 200:
        msg = f"Groq API request failed (HTTP {r.status_code})"
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
    return ans if ans else "ERROR: Empty response from Groq"


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
