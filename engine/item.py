"""The unit of data stored in every index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .distance import Vector


@dataclass
class Item:
    id: int
    embedding: Vector = field(default_factory=list)
    metadata: str = ""
    category: str = ""
    title: Optional[str] = None
    text: Optional[str] = None

    def vector_dict(self) -> dict:
        return {
            "id": self.id,
            "metadata": self.metadata,
            "category": self.category,
            "embedding": self.embedding,
        }
