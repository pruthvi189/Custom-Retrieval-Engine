"""Chunking for RAG document store.

Splits long text into overlapping chunks so each piece stays small enough to
embed and stays coherent. Step is chunk_words - overlap_words.
"""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks
