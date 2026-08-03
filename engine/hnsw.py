"""HNSW - Hierarchical Navigable Small World graph.

Each node lives on one or more layers (fewer nodes per layer the higher you
go); search starts at the top and works down, so it narrows in fast then
refines. Ported faithfully from the JS original: same M / efBuild / mL, same
insert/knn/remove/reset flow, dict-backed adjacency for insertion order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .distance import Vector
from .heaps import Entry, MinHeap, MaxHeap


@dataclass
class HNSWNode:
    item: object
    max_lyr: int
    nbrs: list[list[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.G: dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1

    def rand_level(self) -> int:
        u = random.random()
        if u <= 0:
            return 0
        return math.floor(-math.log(u) * self.mL)

    def search_layer(self, q: Vector, ep: int, ef: int, layer: int, dist) -> list[Entry]:
        vis: set[int] = set()
        cands = MinHeap()
        found = MaxHeap()
        if ep not in self.G:
            return []
        d0 = dist(q, self.G[ep].item.embedding)
        vis.add(ep)
        cands.push(Entry(d=d0, id=ep))
        found.push(Entry(d=d0, id=ep))
        while cands.size:
            c = cands.pop()
            if found.size >= ef and c.d > found.peek().d:
                break
            node = self.G.get(c.id)
            if node is None or layer >= len(node.nbrs):
                continue
            for nid in node.nbrs[layer]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.embedding)
                if found.size < ef or nd < found.peek().d:
                    cands.push(Entry(d=nd, id=nid))
                    found.push(Entry(d=nd, id=nid))
                    if found.size > ef:
                        found.pop()
        res = []
        while found.size:
            res.append(found.pop())
        res.sort(key=lambda e: e.d)
        return res

    def select_nbrs(self, cands: list[Entry], max_m: int) -> list[int]:
        return [x.id for x in cands[:max_m]]

    def insert(self, item, dist) -> None:
        id = item.id
        lvl = self.rand_level()
        nbrs = [[] for _ in range(lvl + 1)]
        self.G[id] = HNSWNode(item=item, max_lyr=lvl, nbrs=nbrs)

        if self.entry_pt == -1:
            self.entry_pt = id
            self.top_layer = lvl
            return

        # first glide down through layers above the node's level
        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item.embedding, ep, 1, lc, dist)
                if W:
                    ep = W[0].id
        # then connect it into each layer it lives on
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self.search_layer(item.embedding, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nd = self.G[nid]
                while len(nd.nbrs) <= lc:
                    nd.nbrs.append([])
                conn = nd.nbrs[lc]
                conn.append(id)
                # keep the neighbor list bounded - prune to the closest max_m
                if len(conn) > max_m:
                    ds = sorted(
                        ((dist(nd.item.embedding, self.G[c].item.embedding), c) for c in conn),
                        key=lambda t: t[0],
                    )
                    nd.nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0].id
        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id

    def knn(self, q: Vector, k: int, ef: int, dist) -> list[Entry]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0].id
        W = self.search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int) -> None:
        if id not in self.G:
            return
        # sever every back-reference to this node, then drop it
        for nd in self.G.values():
            for layer in nd.nbrs:
                try:
                    layer.remove(id)
                except ValueError:
                    pass
        if self.entry_pt == id:
            self.entry_pt = -1
            for nid in self.G.keys():
                if nid != id:
                    self.entry_pt = nid
                    break
        del self.G[id]

    def reset(self) -> None:
        self.G = {}
        self.top_layer = -1
        self.entry_pt = -1

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id, nd in self.G.items():
            nodes.append(
                {
                    "id": id,
                    "metadata": nd.item.metadata,
                    "category": nd.item.category,
                    "maxLyr": nd.max_lyr,
                }
            )
            for lc in range(0, min(nd.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }
