"""Custom Retrieval Engine - vector search core.

Started as a port of a C++ experiment (via a Node.js intermediate) and ended
up its own thing: three search indexes (brute force, kd-tree, HNSW) plus a
chunked document store for RAG. Libraries own the pieces that have good ones -
stdlib ``heapq`` for the heaps, ``scipy.spatial.cKDTree`` for euclidean and
manhattan k-NN, numpy for the distance metrics - while HNSW and the cosine
kd-tree stay hand-written.
"""

DIMS = 16

__version__ = "2.0.0"

from .distance import euclidean, cosine, manhattan, get_dist_fn
from .heaps import Entry, MinHeap, MaxHeap
from .kdtree import KDTree
from .hnsw import HNSW
from .chunking import chunk_text
from .text_features import graph_embedding
from .demo import DEMO, load_demo
from .vectordb import VectorDB, DocumentDB, BruteForce

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
