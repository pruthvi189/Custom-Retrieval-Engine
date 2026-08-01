// Core vector engine. Started as a port of a C++ experiment and ended up
// its own thing — three search indexes (brute force, kd-tree, HNSW) plus a
// chunked document store for RAG.

const DIMS = 16;

// Distance metrics. Euclidean and manhattan are plain L2/L1; cosine is
// returned as a "distance" too (1 - similarity) so the same code path
// works for all three.
function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  // degenerate vectors: treat as maximally dissimilar
  if (na < 1e-9 || nb < 1e-9) return 1;
  return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function manhattan(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

function getDistFn(m) {
  if (m === 'cosine') return cosine;
  if (m === 'manhattan') return manhattan;
  return euclidean;
}

// Binary heaps on {d, id} entries. Used to keep the top-k candidates
// during tree searches without sorting the whole dataset.
class MinHeap {
  constructor() { this.a = []; }
  push(x) {
    const a = this.a; a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[i].d >= a[p].d) break;
      [a[i], a[p]] = [a[p], a[i]]; i = p;
    }
  }
  pop() {
    const a = this.a;
    if (!a.length) return null;
    const top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l].d < a[m].d) m = l;
        if (r < a.length && a[r].d < a[m].d) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
  peek() { return this.a[0] || null; }
  get size() { return this.a.length; }
}

class MaxHeap {
  constructor() { this.a = []; }
  push(x) {
    const a = this.a; a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[i].d <= a[p].d) break;
      [a[i], a[p]] = [a[p], a[i]]; i = p;
    }
  }
  pop() {
    const a = this.a;
    if (!a.length) return null;
    const top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l].d > a[m].d) m = l;
        if (r < a.length && a[r].d > a[m].d) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
  peek() { return this.a[0] || null; }
  get size() { return this.a.length; }
}

// Brute force — the baseline every other index gets compared against.
// Just stores everything and does a linear scan per query.
class BruteForce {
  constructor() { this.items = []; }
  insert(v) { this.items.push(v); }
  knn(q, k, dist) {
    const r = [];
    for (const v of this.items) r.push({ d: dist(q, v.embedding), id: v.id });
    r.sort((a, b) => a.d - b.d);
    return r.slice(0, k);
  }
  remove(id) {
    this.items = this.items.filter(v => v.id !== id);
  }
}

// kd-tree. Splits on one dimension per level (axis cycles through 0..dims),
// so lookups are O(log n) in the best case. Deletions are handled by
// rebuilding the whole tree from scratch — fine at this scale.
class KDNode {
  constructor(v) { this.item = v; this.left = null; this.right = null; }
}

class KDTree {
  constructor(dims) { this.dims = dims; this.root = null; }
  insert(n, v, d) {
    if (!n) return new KDNode(v);
    const ax = d % this.dims;
    if (v.embedding[ax] < n.item.embedding[ax]) n.left = this.insert(n.left, v, d + 1);
    else n.right = this.insert(n.right, v, d + 1);
    return n;
  }
  insertItem(item) {
    this.root = this.insert(this.root, item, 0);
  }
  knnRec(n, q, k, d, dist, heap) {
    if (!n) return;
    const dn = dist(q, n.item.embedding);
    if (heap.size < k || dn < heap.peek().d) {
      heap.push({ d: dn, id: n.item.id });
      if (heap.size > k) heap.pop();
    }
    // descend the side the query falls on, then backtrack if the hyperplane
    // is close enough that a better match could live on the far side
    const ax = d % this.dims;
    const diff = q[ax] - n.item.embedding[ax];
    const closer = diff < 0 ? n.left : n.right;
    const farther = diff < 0 ? n.right : n.left;
    this.knnRec(closer, q, k, d + 1, dist, heap);
    if (heap.size < k || Math.abs(diff) < heap.peek().d)
      this.knnRec(farther, q, k, d + 1, dist, heap);
  }
  knn(q, k, dist) {
    const heap = new MaxHeap();
    this.knnRec(this.root, q, k, 0, dist, heap);
    const r = [];
    while (heap.size) r.push(heap.pop());
    r.sort((a, b) => a.d - b.d);
    return r;
  }
  rebuild(items) {
    this.root = null;
    for (const v of items) this.root = this.insert(this.root, v, 0);
  }
}

// HNSW — Hierarchical Navigable Small World graph. Each node lives on one
// or more layers (fewer nodes per layer the higher you go); search starts
// at the top and works down, so it narrows in fast then refines.
class HNSW {
  constructor(m = 16, efBuild = 200) {
    this.M = m;
    this.M0 = 2 * m;
    this.ef_build = efBuild;
    this.mL = 1 / Math.log(m);
    this.G = new Map();
    this.topLayer = -1;
    this.entryPt = -1;
  }
  randLevel() {
    const u = Math.random();
    if (u <= 0) return 0;
    return Math.floor(-Math.log(u) * this.mL);
  }
  searchLayer(q, ep, ef, lyr, dist) {
    const vis = new Set();
    const cands = new MinHeap(), found = new MaxHeap();
    if (!this.G.has(ep)) return [];
    const d0 = dist(q, this.G.get(ep).item.embedding);
    vis.add(ep); cands.push({ d: d0, id: ep }); found.push({ d: d0, id: ep });
    while (cands.size) {
      const c = cands.pop();
      if (found.size >= ef && c.d > found.peek().d) break;
      const node = this.G.get(c.id);
      if (!node || lyr >= node.nbrs.length) continue;
      for (const nid of node.nbrs[lyr]) {
        if (vis.has(nid) || !this.G.has(nid)) continue;
        vis.add(nid);
        const nd = dist(q, this.G.get(nid).item.embedding);
        if (found.size < ef || nd < found.peek().d) {
          cands.push({ d: nd, id: nid });
          found.push({ d: nd, id: nid });
          if (found.size > ef) found.pop();
        }
      }
    }
    const res = [];
    while (found.size) res.push(found.pop());
    res.sort((a, b) => a.d - b.d);
    return res;
  }
  selectNbrs(cands, maxM) {
    return cands.slice(0, maxM).map(x => x.id);
  }
  insert(item, dist) {
    const id = item.id, lvl = this.randLevel();
    const nbrs = [];
    for (let i = 0; i <= lvl; i++) nbrs.push([]);
    this.G.set(id, { item, maxLyr: lvl, nbrs });

    if (this.entryPt === -1) { this.entryPt = id; this.topLayer = lvl; return; }

    // first glide down through layers above the node's level
    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > lvl; lc--) {
      if (lc < this.G.get(ep).nbrs.length) {
        const W = this.searchLayer(item.embedding, ep, 1, lc, dist);
        if (W.length) ep = W[0].id;
      }
    }
    // then connect it into each layer it lives on
    for (let lc = Math.min(this.topLayer, lvl); lc >= 0; lc--) {
      const W = this.searchLayer(item.embedding, ep, this.ef_build, lc, dist);
      const maxM = (lc === 0) ? this.M0 : this.M;
      const sel = this.selectNbrs(W, maxM);
      this.G.get(id).nbrs[lc] = sel;

      for (const nid of sel) {
        if (!this.G.has(nid)) continue;
        const nd = this.G.get(nid);
        while (nd.nbrs.length <= lc) nd.nbrs.push([]);
        const conn = nd.nbrs[lc];
        conn.push(id);
        // keep the neighbor list bounded — prune to the closest maxM
        if (conn.length > maxM) {
          const ds = conn.map(c => ({ d: dist(nd.item.embedding, this.G.get(c).item.embedding), id: c }));
          ds.sort((a, b) => a.d - b.d);
          nd.nbrs[lc] = ds.slice(0, maxM).map(x => x.id);
        }
      }
      if (W.length) ep = W[0].id;
    }
    if (lvl > this.topLayer) { this.topLayer = lvl; this.entryPt = id; }
  }
  knn(q, k, ef, dist) {
    if (this.entryPt === -1) return [];
    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > 0; lc--) {
      if (lc < this.G.get(ep).nbrs.length) {
        const W = this.searchLayer(q, ep, 1, lc, dist);
        if (W.length) ep = W[0].id;
      }
    }
    const W = this.searchLayer(q, ep, Math.max(ef, k), 0, dist);
    return W.slice(0, k);
  }
  remove(id) {
    if (!this.G.has(id)) return;
    // sever every back-reference to this node, then drop it
    for (const nd of this.G.values())
      for (const layer of nd.nbrs)
        layer.splice(layer.indexOf(id), 1);
    if (this.entryPt === id) {
      this.entryPt = -1;
      for (const nid of this.G.keys()) if (nid !== id) { this.entryPt = nid; break; }
    }
    this.G.delete(id);
  }
  reset() { this.G = new Map(); this.topLayer = -1; this.entryPt = -1; }
  // used by /api/hnsw-info to render the graph in the browser
  getInfo() {
    const maxL = Math.max(this.topLayer + 1, 1);
    const nodesPerLayer = new Array(maxL).fill(0);
    const edgesPerLayer = new Array(maxL).fill(0);
    const nodes = [], edges = [];
    for (const [id, nd] of this.G) {
      nodes.push({ id, metadata: nd.item.metadata, category: nd.item.category, maxLyr: nd.maxLyr });
      for (let lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
        nodesPerLayer[lc]++;
        if (lc < nd.nbrs.length)
          for (const nid of nd.nbrs[lc])
            if (id < nid) {
              edgesPerLayer[lc]++;
              edges.push({ src: id, dst: nid, lyr: lc });
            }
      }
    }
    return { topLayer: this.topLayer, nodeCount: this.G.size, nodesPerLayer, edgesPerLayer, nodes, edges };
  }
}

// VectorDB — the 16D demo index. Keeps all three algorithms in sync so any
// endpoint can compare them side by side. Items live in one source of truth
// (`store`), and each index gets a copy of it.
class VectorDB {
  constructor(dims) {
    this.dims = dims;
    this.store = new Map();
    this.bf = new BruteForce();
    this.kdt = new KDTree(dims);
    this.hnsw = new HNSW(16, 200);
    this.nextId = 1;
  }
  insert(meta, cat, emb) {
    const id = this.nextId++;
    this.insertRaw({ id, metadata: meta, category: cat, embedding: emb });
    return id;
  }
  insertRaw(item) {
    this.store.set(item.id, item);
    this.bf.insert(item);
    this.kdt.insertItem(item);
    this.hnsw.insert(item, cosine);
    if (item.id >= this.nextId) this.nextId = item.id + 1;
  }
  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.bf.remove(id);
    this.hnsw.remove(id);
    this.kdt.rebuild([...this.store.values()]);
    return true;
  }
  reset() {
    this.store = new Map();
    this.bf = new BruteForce();
    this.kdt = new KDTree(this.dims);
    this.hnsw = new HNSW(16, 200);
    this.nextId = 1;
  }
  search(q, k, metric, algo) {
    const dfn = getDistFn(metric);
    const t0 = process.hrtime.bigint();
    let raw;
    if (algo === 'bruteforce') raw = this.bf.knn(q, k, dfn);
    else if (algo === 'kdtree') raw = this.kdt.knn(q, k, dfn);
    else raw = this.hnsw.knn(q, k, 50, dfn);
    const us = Number(process.hrtime.bigint() - t0) / 1000;
    const hits = [];
    for (const { d, id } of raw) {
      const it = this.store.get(id);
      if (it) hits.push({ id, metadata: it.metadata, category: it.category, embedding: it.embedding, distance: d });
    }
    return { hits, us, algo, metric };
  }
  benchmark(q, k, metric) {
    const dfn = getDistFn(metric);
    const time = fn => {
      const t = process.hrtime.bigint();
      fn();
      return Number(process.hrtime.bigint() - t) / 1000;
    };
    return {
      bruteforceUs: time(() => this.bf.knn(q, k, dfn)),
      kdtreeUs: time(() => this.kdt.knn(q, k, dfn)),
      hnswUs: time(() => this.hnsw.knn(q, k, 50, dfn)),
      itemCount: this.store.size
    };
  }
  all() { return [...this.store.values()]; }
  hnswInfo() { return this.hnsw.getInfo(); }
  size() { return this.store.size; }
}

// DocumentDB — chunks of real text embedded with OpenRouter (1536D). Only
// brute-force search for now; doc sets here are small enough that it's fine.
class DocumentDB {
  constructor() {
    this.store = new Map();
    this.bf = new BruteForce();
    this.nextId = 1;
    this.dims = 0;
  }
  insert(title, text, emb) {
    if (!this.dims) this.dims = emb.length;
    const id = this.nextId++;
    const item = { id, title, text, embedding: emb };
    this.store.set(id, item);
    this.bf.insert(item);
    return id;
  }
  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.bf.remove(id);
    return true;
  }
  search(q, k, maxDist = 0.7) {
    if (!this.store.size) return [];
    const raw = this.bf.knn(q, k, cosine);
    const out = [];
    for (const { d, id } of raw) {
      const it = this.store.get(id);
      if (it && d <= maxDist) out.push({ distance: d, item: it });
    }
    return out;
  }
  all() { return [...this.store.values()]; }
  size() { return this.store.size; }
  getDims() { return this.dims; }
}

// Splits long text into overlapping chunks so each piece stays small enough
// to embed and stays coherent. Step is chunkWords - overlapWords.
function chunkText(text, chunkWords = 250, overlapWords = 30) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  if (words.length <= chunkWords) return [text];
  const chunks = [];
  const step = chunkWords - overlapWords;
  for (let i = 0; i < words.length; i += step) {
    const end = Math.min(i + chunkWords, words.length);
    chunks.push(words.slice(i, end).join(' '));
    if (end === words.length) break;
  }
  return chunks;
}

// Demo data — 16D vectors grouped by category (cs, math, food, sports).
// The first four dimensions encode the category strongly enough that a
// category filter is just a nearest-neighbour query.
const DEMO = [
  ["Linked List: nodes connected by pointers", "cs",
    [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]],
  ["Binary Search Tree: O(log n) search and insert", "cs",
    [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]],
  ["Dynamic Programming: memoization overlapping subproblems", "cs",
    [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]],
  ["Graph BFS and DFS: breadth and depth first traversal", "cs",
    [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]],
  ["Hash Table: O(1) lookup with collision chaining", "cs",
    [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]],
  ["Calculus: derivatives integrals and limits", "math",
    [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]],
  ["Linear Algebra: matrices eigenvalues eigenvectors", "math",
    [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]],
  ["Probability: distributions random variables Bayes theorem", "math",
    [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]],
  ["Number Theory: primes modular arithmetic RSA cryptography", "math",
    [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]],
  ["Combinatorics: permutations combinations generating functions", "math",
    [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]],
  ["Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
    [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]],
  ["Sushi: vinegared rice raw fish and nori rolls", "food",
    [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]],
  ["Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
    [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]],
  ["Tacos: corn tortillas with carnitas salsa and cilantro", "food",
    [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]],
  ["Croissant: laminated pastry with buttery flaky layers", "food",
    [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]],
  ["Basketball: fast-paced shooting dribbling slam dunks", "sports",
    [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]],
  ["Football: tackles touchdowns field goals and strategy", "sports",
    [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]],
  ["Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
    [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]],
  ["Chess: openings endgames tactics strategic board game", "sports",
    [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]],
  ["Swimming: butterfly freestyle backstroke Olympic competition", "sports",
    [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]]
];

function loadDemo(vdb) {
  for (const [meta, cat, emb] of DEMO) vdb.insert(meta, cat, emb);
}

module.exports = {
  DIMS, euclidean, cosine, manhattan, getDistFn,
  BruteForce, KDTree, HNSW, VectorDB, DocumentDB,
  chunkText, DEMO, loadDemo
};
