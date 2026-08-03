"""Deterministic keyword features for the 16D visualizer graph.

Keyword buckets mirror the ones the frontend uses so a web-fetched article
lands in the same region of the 16D graph as a manually inserted document
about that topic.
"""

from __future__ import annotations

KW: dict[str, list[str]] = {
    "cs": [
        "algorithm", "data", "tree", "graph", "array", "linked", "hash", "stack", "queue",
        "sort", "binary", "dynamic", "programming", "recursion", "complexity", "pointer",
        "node", "search", "insert", "bfs", "dfs", "heap", "trie", "database", "index",
        "query", "sql", "vector", "embedding", "semantic", "network", "distributed",
        "cache", "memory", "runtime", "thread", "process", "async", "library", "framework",
    ],
    "math": [
        "calculus", "matrix", "probability", "theorem", "integral", "derivative", "linear",
        "algebra", "equation", "function", "prime", "modular", "combinatorics", "permutation",
        "eigenvalue", "statistics", "proof", "geometry", "trig", "logarithm", "limit",
        "sequence", "series", "fraction", "ratio",
    ],
    "food": [
        "food", "pizza", "sushi", "ramen", "pasta", "recipe", "cook", "eat", "restaurant",
        "dish", "ingredient", "flavor", "spice", "noodle", "bread", "croissant", "taco",
        "fish", "rice", "soup", "biryani", "curry", "kebab", "tikka", "samosa", "dosa",
        "naan", "paneer", "masala", "burger", "fries", "steak", "chicken", "beef", "pork",
        "lamb", "cheese", "egg", "meat", "grill", "barbecue", "roast", "fried", "salad",
        "sandwich", "pancake", "waffle", "cake", "cookie", "pie", "pastry", "chocolate",
        "dessert", "cream", "coffee", "tea", "wine", "beer", "juice", "tomato", "onion",
        "garlic", "mango", "banana",
    ],
    "sports": [
        "sport", "basketball", "football", "tennis", "chess", "swim", "game", "play",
        "score", "team", "athlete", "competition", "match", "tournament", "olympic",
        "dribble", "tackle", "serve", "soccer", "cricket", "hockey", "golf", "boxing",
        "wrestling", "cycling", "running", "marathon", "yoga", "gym", "fitness", "baseball",
        "volleyball", "rugby", "badminton",
    ],
}


def graph_embedding(text: str) -> list[float]:
    """Deterministic 16D embedding for a graph point (no jitter - the
    frontend adds ±0.02 jitter at render time)."""
    t = text.lower()
    ws = [w for w in t.split()]
    s = {"cs": 0, "math": 0, "food": 0, "sports": 0}
    for w in ws:
        for cat, kws in KW.items():
            for kw in kws:
                if kw in w or w.startswith(kw):
                    s[cat] += 0.35
                    break
    mx = max(list(s.values()) + [0.01])
    n = lambda v: min(v / mx * 0.88, 0.94)
    emb = [0.08] * 16

    def fill(i: int, score: float) -> None:
        if score < 0.01:
            return
        b = n(score)
        emb[i] = max(0.05, b)
        emb[i + 1] = max(0.05, b)
        emb[i + 2] = max(0.05, b * 0.92)
        emb[i + 3] = max(0.05, b * 0.87)

    fill(0, s["cs"])
    fill(4, s["math"])
    fill(8, s["food"])
    fill(12, s["sports"])
    return emb
