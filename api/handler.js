// Single Vercel function that routes every /api/* path to the store layer.
// One handler keeps cold starts (and deploys) simple instead of one
// serverless function per endpoint.

const { json } = require('./_lib/http');
const store = require('./_lib/store');
const { DIMS } = require('./_lib/core');

module.exports = async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.statusCode = 204;
    return res.end();
  }

  try {
    const url = new URL(req.url, 'http://localhost');
    const path = url.pathname.replace(/\/+$/, '');
    const query = Object.fromEntries(url.searchParams);
    // Vercel already parses JSON bodies, so guard against double-parsing
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : (req.body || {});
    const send = (status, data) => json(res, status, data);

    if (path === '/api/items' && req.method === 'GET') {
      return send(200, await store.items());
    }

    if (path === '/api/search' && req.method === 'GET') {
      const q = (query.v || '').split(',').map(Number).filter(n => Number.isFinite(n));
      if (q.length !== DIMS) return send(400, { error: 'need ' + DIMS + 'D vector' });
      const k = parseInt(query.k, 10) || 5;
      const metric = query.metric || 'cosine';
      const algo = query.algo || 'hnsw';
      const out = await store.searchItems(q, k, metric, algo);
      return send(200, {
        results: out.hits.map(h => ({ id: h.id, metadata: h.metadata, category: h.category, distance: Number(h.distance.toFixed(6)), embedding: h.embedding })),
        latencyUs: Math.round(out.us),
        algo: out.algo,
        metric: out.metric
      });
    }

    if (path === '/api/insert' && req.method === 'POST') {
      const meta = (body.metadata || '').trim();
      const cat = (body.category || '').trim();
      const emb = Array.isArray(body.embedding) ? body.embedding.map(Number) : [];
      if (!meta || emb.length !== DIMS) return send(400, { error: 'invalid body' });
      return send(200, { id: await store.insertItem(meta, cat, emb) });
    }

    const delMatch = path.match(/^\/api\/delete\/(\d+)$/);
    if (delMatch && req.method === 'DELETE') {
      return send(200, { ok: await store.deleteItem(parseInt(delMatch[1], 10)) });
    }

    if (path === '/api/benchmark' && req.method === 'GET') {
      const q = (query.v || '').split(',').map(Number).filter(n => Number.isFinite(n));
      if (q.length !== DIMS) return send(400, { error: 'need ' + DIMS + 'D vector' });
      const k = parseInt(query.k, 10) || 5;
      const metric = query.metric || 'cosine';
      const b = await store.benchmark(q, k, metric);
      return send(200, { bruteforceUs: Math.round(b.bruteforceUs), kdtreeUs: Math.round(b.kdtreeUs), hnswUs: Math.round(b.hnswUs), itemCount: b.itemCount });
    }

    if (path === '/api/hnsw-info' && req.method === 'GET') {
      return send(200, await store.hnswInfo());
    }

    if (path === '/api/status' && req.method === 'GET') {
      return send(200, await store.status());
    }

    if (path === '/api/stats' && req.method === 'GET') {
      await store.items();
      return send(200, store.stats());
    }

    if (path === '/api/agent/ingest' && req.method === 'POST') {
      const topic = (body.topic || body.question || '').trim();
      const maxArticles = parseInt(body.maxArticles, 10) || 1;
      if (!topic) return send(400, { error: 'need topic' });
      const result = await store.webIngest(topic, maxArticles);
      if (result.error) return send(400, { error: result.error });
      return send(200, result);
    }

    if (path === '/api/doc/insert' && req.method === 'POST') {
      const title = (body.title || '').trim();
      const text = (body.text || '').trim();
      if (!title || !text) return send(400, { error: 'need title and text' });
      const result = await store.docInsert(title, text);
      if (result.error) return send(400, { error: result.error });
      return send(200, { ids: result.ids, chunks: result.chunks, dims: result.dims });
    }

    if (path === '/api/doc/list' && req.method === 'GET') {
      return send(200, await store.docList());
    }

    if (path === '/api/doc/search' && req.method === 'POST') {
      const question = (body.question || '').trim();
      const k = parseInt(body.k, 10) || 3;
      if (!question) return send(400, { error: 'need question' });
      const result = await store.docSearch(question, k);
      if (result.error) return send(400, { error: result.error });
      return send(200, { contexts: result.contexts });
    }

    if (path === '/api/doc/ask' && req.method === 'POST') {
      const question = (body.question || '').trim();
      const k = parseInt(body.k, 10) || 3;
      if (!question) return send(400, { error: 'need question' });
      const result = await store.docAsk(question, k);
      if (result.error) return send(400, { error: result.error });
      return send(200, result);
    }

    const delDocMatch = path.match(/^\/api\/doc\/delete\/(\d+)$/);
    if (delDocMatch && req.method === 'DELETE') {
      return send(200, { ok: await store.docDelete(parseInt(delDocMatch[1], 10)) });
    }

    send(404, { error: 'not found' });

  } catch (e) {
    json(res, 500, { error: e.message });
  }
};
