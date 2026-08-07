"""Tool registry with normalized interfaces for the agent.

Each tool returns a consistent structure:
{
    "tool": "tool_name",
    "query": "original query",
    "results": [...],
    "sources": [...]
}
"""

from __future__ import annotations

import httpx
import os
from typing import Any

import api.store as store
from api.background import enqueue_ingestion


# ---- Normalized result structure --------------------------------------------------------

class ToolResult:
    """Normalized tool result."""

    def __init__(
        self,
        tool: str,
        query: str,
        results: list[dict],
        sources: list[dict],
        error: str | None = None,
    ):
        self.tool = tool
        self.query = query
        self.results = results
        self.sources = sources
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tool": self.tool,
            "query": self.query,
            "results": self.results,
            "sources": self.sources,
        }
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_error(cls, tool: str, query: str, error: str) -> "ToolResult":
        return cls(tool=tool, query=query, results=[], sources=[], error=error)


# ---- Tool implementations ---------------------------------------------------------------

_WIKI_UA = "CustomRetrievalEngine-RAG/1.0"
_TAVILY_KEY = os.environ.get("TAVILY_API_KEY")


def doc_search(query: str, k: int = 5) -> ToolResult:
    """Search local document chunks via semantic search."""
    try:
        result = store.doc_search(query, k)
        if result.get("error"):
            return ToolResult.from_error("doc_search", query, result["error"])

        contexts = result.get("contexts", [])
        results = []
        sources = []
        for ctx in contexts:
            results.append({
                "id": ctx.get("id"),
                "title": ctx.get("title"),
                "distance": ctx.get("distance"),
            })
            sources.append({
                "type": "local_doc",
                "id": ctx.get("id"),
                "title": ctx.get("title"),
                "distance": ctx.get("distance"),
            })

        return ToolResult(
            tool="doc_search",
            query=query,
            results=results,
            sources=sources,
        )
    except Exception as e:
        return ToolResult.from_error("doc_search", query, str(e))


def wiki_search(query: str, max_articles: int = 2, background: bool = False) -> ToolResult:
    """Search and ingest Wikipedia articles.
    
    Args:
        query: Search topic
        max_articles: Maximum articles to ingest
        background: If True, enqueue ingestion and return immediately
    """
    if background:
        enqueue_ingestion(query, max_articles)
        return ToolResult(
            tool="wiki_search",
            query=query,
            results=[{"status": "queued", "topic": query, "max_articles": max_articles}],
            sources=[{"type": "wikipedia", "title": query, "status": "queued"}],
        )
    try:
        result = store.web_ingest(query, max_articles)
        if result.get("error"):
            return ToolResult.from_error("wiki_search", query, result["error"])

        added = result.get("added", [])
        results = []
        sources = []
        for a in added:
            results.append({
                "title": a.get("title"),
                "chunks": a.get("chunks"),
                "stored": a.get("stored"),
            })
            sources.append({
                "type": "wikipedia",
                "title": a.get("title"),
                "chunks": a.get("chunks"),
            })

        return ToolResult(
            tool="wiki_search",
            query=query,
            results=results,
            sources=sources,
        )
    except Exception as e:
        return ToolResult.from_error("wiki_search", query, str(e))


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web via Tavily API. Returns temporary context only - does not persist."""
    if not _TAVILY_KEY:
        return ToolResult.from_error(
            "web_search", query, "TAVILY_API_KEY not configured"
        )

    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": _TAVILY_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=30.0,
        )

        if r.status_code != 200:
            return ToolResult.from_error(
                "web_search", query, f"Tavily API error: {r.status_code}"
            )

        data = r.json()
        results = []
        sources = []

        # Include Tavily's direct answer if available
        answer = data.get("answer")
        if answer:
            results.append({"type": "answer", "content": answer})
            sources.append({"type": "web_answer", "content": answer[:200]})

        for item in data.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content", "")[:500],
                "score": item.get("score"),
            })
            sources.append({
                "type": "web",
                "title": item.get("title"),
                "url": item.get("url"),
                "score": item.get("score"),
            })

        return ToolResult(
            tool="web_search",
            query=query,
            results=results,
            sources=sources,
        )
    except httpx.TimeoutException:
        return ToolResult.from_error("web_search", query, "Request timeout")
    except httpx.HTTPError as e:
        return ToolResult.from_error("web_search", query, f"Network error: {e}")
    except Exception as e:
        return ToolResult.from_error("web_search", query, str(e))


# ---- Tool registry ----------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "doc_search": {
        "func": doc_search,
        "description": "Search your local document chunks for relevant context. Use for questions about documents you've already ingested.",
        "params": {"query": "string", "k": "int (default 5)"},
    },
    "wiki_search": {
        "func": wiki_search,
        "description": "Search Wikipedia and permanently ingest articles into your knowledge base. Use for established facts, definitions, historical topics.",
        "params": {"query": "string", "max_articles": "int (default 2)"},
    },
    "web_search": {
        "func": web_search,
        "description": "Search the live web for current information, news, recent developments. Results are temporary context only - not saved to knowledge base.",
        "params": {"query": "string", "max_results": "int (default 5)"},
    },
}


def get_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def get_tool_descriptions() -> str:
    """Format tool descriptions for the planner prompt."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool by name with kwargs."""
    if name not in TOOL_REGISTRY:
        return ToolResult.from_error(name, str(kwargs), f"Unknown tool: {name}")
    try:
        return TOOL_REGISTRY[name]["func"](**kwargs)
    except Exception as e:
        return ToolResult.from_error(name, str(kwargs), f"Execution error: {e}")