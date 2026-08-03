"""Custom Retrieval Engine - pure-Python vector search core.

Started as a port of a C++ experiment (via a Node.js intermediate) and ended
up its own thing: three search indexes (brute force, kd-tree, HNSW) plus a
chunked document store for RAG. All of it implemented by hand in numpy-free
pure Python so the API path stays dependency-light and cold-start fast.

numpy lives only in :mod:`engine.numpy_ops` and is used by the benchmark
scripts and opt-in fast paths - never by the server hot path.
"""

DIMS = 16

__version__ = "2.0.0"

from .distance import euclidean, cosine, manhattan, get_dist_fn
from .heaps import Entry, MinHeap, MaxHeap
from .bruteforce import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW
from .chunking import chunk_text
from .text_features import graph_embedding
from .demo import DEMO, load_demo
from .vectordb import VectorDB, DocumentDB

__all__ = [
    "DIMS",
    "euclidean",
    "cosine",
    "manhattan",
    "get_dist_fn",
    "Entry",
    "MinHeap",
    "MaxHeap",
    "BruteForce",
    "KDTree",
    "HNSW",
    "chunk_text",
    "graph_embedding",
    "DEMO",
    "load_demo",
    "VectorDB",
    "DocumentDB",
]
