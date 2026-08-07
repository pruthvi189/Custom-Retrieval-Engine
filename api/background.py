"""Background task queue for async ingestion.

In production on Vercel, this would be replaced with a proper queue (Redis, SQS, etc.).
"""

from __future__ import annotations

import threading
from typing import Any


class BackgroundTaskQueue:
    """Simple thread-based background task queue for async ingestion."""
    def __init__(self) -> None:
        self._queue: list[tuple[callable, tuple, dict]] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, func: callable, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._queue.append((func, args, kwargs))

    def _run(self) -> None:
        while True:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.pop(0)
            if task:
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception:
                    pass  # Log in production
            threading.Event().wait(0.5)


# Global instance
_background_queue = BackgroundTaskQueue()


def enqueue_ingestion(topic: str, max_articles: int = 1) -> None:
    """Enqueue Wikipedia ingestion to run in background."""
    # Import here to avoid circular imports
    import api.store as store
    _background_queue.enqueue(store.web_ingest, topic, max_articles)