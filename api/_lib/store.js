// Store layer — the in-memory engine with optional Postgres persistence.
// If DATABASE_URL or POSTGRES_URL is set, vectors and doc chunks are
// persisted to a pgvector database and reloaded on each cold start.
// Otherwise everything runs in-memory (which is ephemeral on serverless).

const core = require('./core');
const providers = require('./providers');

const { DIMS, VectorDB, DocumentDB, chunkText, loadDemo } = core;

let pool = null;
let initPromise = null;

const vdb = new VectorDB(DIMS);
const docDB = new DocumentDB();

function dbUrl() {
  return process.env.DATABASE_URL || process.env.POSTGRES_URL || '';
}

// pgvector stores vectors as text like [0.12,0.90,...]
function vecToString(v) {
  return '[' + v.map(x => Number(x).toFixed(6)).join(',') + ']';
}

async function getPool() {
  if (pool) return pool;
  if (!dbUrl()) return null;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      const { Pool } = require('pg');
      const p = new Pool({
        connectionString: dbUrl(),
        ssl: { rejectUnauthorized: false },
        max: 1,
        connectionTimeoutMillis: 8000,
        idleTimeoutMillis: 30000
      });
      await p.query('CREATE EXTENSION IF NOT EXISTS vector');
      await p.query('CREATE TABLE IF NOT EXISTS items (id SERIAL PRIMARY KEY, metadata TEXT NOT NULL, category TEXT NOT NULL, embedding vector(16) NOT NULL)');
      await p.query('CREATE TABLE IF NOT EXISTS doc_chunks (id SERIAL PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL, embedding vector(1536) NOT NULL)');
      pool = p;
      return pool;
    } catch (e) {
      console.error('pg init failed, falling back to in-memory:', e.message);
      pool = null;
      return null;
    }
  })();
  const result = await initPromise;
  initPromise = null;
  return result;
}

async function loadItems() {
  const p = await getPool();
  if (!p) return;
  try {
    const r = await p.query('SELECT id, metadata, category, embedding::text AS e FROM items ORDER BY id');
    vdb.reset();
    for (const row of r.rows) {
      vdb.insertRaw({ id: row.id, metadata: row.metadata, category: row.category, embedding: JSON.parse(row.e) });
    }
  } catch (e) {
    console.error('loadItems failed:', e.message);
  }
}

// Make sure the demo index has data before the first request. With Postgres
// we seed once and then always load from it; in-memory mode just seeds on
// first use.
async function ensureDemo() {
  const p = await getPool();
  if (p) {
    const { rows } = await p.query('SELECT COUNT(*)::int AS n FROM items');
    if (rows[0].n === 0) {
      for (const [meta, cat, emb] of core.DEMO) {
        await p.query('INSERT INTO items (metadata, category, embedding) VALUES ($1, $2, $3::vector)', [meta, cat, vecToString(emb)]);
      }
    }
    await loadItems();
  } else if (vdb.size() === 0) {
    loadDemo(vdb);
  }
}

// items / search / insert / delete

async function items() {
  await ensureDemo();
  return vdb.all();
}

async function searchItems(q, k, metric, algo) {
  await ensureDemo();
  await loadItems();
  return vdb.search(q, k, metric, algo);
}

async function insertItem(meta, cat, emb) {
  await ensureDemo();
  const p = await getPool();
  if (p) {
    const r = await p.query('INSERT INTO items (metadata, category, embedding) VALUES ($1, $2, $3::vector) RETURNING id', [meta, cat, vecToString(emb)]);
    const id = r.rows[0].id;
    vdb.insertRaw({ id, metadata: meta, category: cat, embedding: emb });
    return id;
  }
  return vdb.insert(meta, cat, emb);
}

async function deleteItem(id) {
  await ensureDemo();
  const p = await getPool();
  if (p) {
    await p.query('DELETE FROM items WHERE id = $1', [id]);
  }
  return vdb.remove(id);
}

async function benchmark(q, k, metric) {
  await ensureDemo();
  await loadItems();
  return vdb.benchmark(q, k, metric);
}

async function hnswInfo() {
  await ensureDemo();
  await loadItems();
  return vdb.hnswInfo();
}

// documents + RAG

async function docInsert(title, text) {
  await ensureDemo();
  const chunks = chunkText(text, 250, 30);
  const ids = [];
  for (let i = 0; i < chunks.length; i++) {
    const emb = await providers.embedOne(chunks[i]);
    if (!emb) return { error: 'Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var.' };
    const chunkTitle = chunks.length > 1 ? title + ' [' + (i + 1) + '/' + chunks.length + ']' : title;
    ids.push(await storeChunk(chunkTitle, chunks[i], emb));
  }
  const dims = (await getPool()) ? 1536 : docDB.getDims();
  return { ids, chunks: chunks.length, dims };
}

async function docList() {
  await ensureDemo();
  const p = await getPool();
  const rows = p ? (await p.query('SELECT id, title, text FROM doc_chunks ORDER BY id')).rows
                 : docDB.all().map(d => ({ id: d.id, title: d.title, text: d.text }));
  return rows.map(d => {
    const preview = d.text.length > 120 ? d.text.slice(0, 120) + '…' : d.text;
    const words = d.text.trim().split(/\s+/).filter(Boolean).length;
    return { id: d.id, title: d.title, preview, words };
  });
}

async function docDelete(id) {
  await ensureDemo();
  const p = await getPool();
  if (p) {
    await p.query('DELETE FROM doc_chunks WHERE id = $1', [id]);
  }
  return docDB.remove(id);
}

async function docSearch(question, k) {
  await ensureDemo();
  const qEmb = await providers.embedOne(question);
  if (!qEmb) return { error: 'Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var.' };
  const p = await getPool();
  if (p) {
    const { rows } = await p.query(
      'SELECT id, title, embedding <=> $1::vector AS distance FROM doc_chunks ORDER BY embedding <=> $1::vector LIMIT $2',
      [vecToString(qEmb), k]);
    const contexts = rows
      .filter(r => r.distance <= 0.7)
      .map(r => ({ id: r.id, title: r.title, distance: r.distance }));
    return { contexts };
  }
  const hits = docDB.search(qEmb, k, 0.7);
  return { contexts: hits.map(h => ({ id: h.item.id, title: h.item.title, distance: h.distance })) };
}

// Retrieval is grounded: the LLM only ever sees chunks that actually matched
// the query. If the database is empty or nothing clears the similarity
// threshold, we say so instead of letting the model guess.
async function docAsk(question, k) {
  await ensureDemo();
  const qEmb = await providers.embedOne(question);
  if (!qEmb) return { error: 'Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var.' };
  const p = await getPool();
  const docCount = p ? (await p.query('SELECT COUNT(*)::int AS n FROM doc_chunks')).rows[0].n : docDB.size();
  let hits;
  if (p) {
    const { rows } = await p.query(
      'SELECT id, title, text, embedding <=> $1::vector AS distance FROM doc_chunks ORDER BY embedding <=> $1::vector LIMIT $2',
      [vecToString(qEmb), k]);
    hits = rows.filter(r => r.distance <= 0.7).map(r => ({ id: r.id, title: r.title, text: r.text, distance: r.distance }));
  } else {
    hits = docDB.search(qEmb, k, 0.7).map(h => ({ id: h.item.id, title: h.item.title, text: h.item.text, distance: h.distance }));
  }

  if (docCount === 0) {
    return {
      answer: 'No documents in the database yet. Use "Search the web for more" below to fetch knowledge about this topic.',
      model: providers.GEN_MODEL, contexts: [], docCount, notFound: true
    };
  }

  if (hits.length === 0) {
    return {
      answer: 'Not found in your documents.',
      model: providers.GEN_MODEL, contexts: [], docCount, notFound: true
    };
  }

  const ctx = hits.map((h, i) => '[' + (i + 1) + '] ' + h.title + ':\n' + h.text + '\n\n').join('');
  const prompt =
    'You are a RAG assistant grounded strictly in the provided context. ' +
    "Answer the user's question using ONLY the context below. " +
    'If the context does not contain enough information to answer, reply exactly: Not found in your documents. ' +
    'Do not use your own general knowledge. Do not mention the context.\n\n' +
    'Context:\n' + ctx +
    'Question: ' + question + '\n\n' +
    'Answer:';

  const answer = await providers.generate(prompt);
  const notFound = /not found in your documents/i.test(answer);
  return { answer, model: providers.GEN_MODEL, contexts: hits, docCount, notFound };
}

// web ingest (Wikipedia, no API key)

async function wikiSearch(topic) {
  const url = 'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=' + encodeURIComponent(topic) +
    '&format=json&srlimit=3';
  const r = await fetch(url, { headers: { 'User-Agent': 'VectorDB-RAG/1.0' } });
  const d = await r.json();
  return (d.query && d.query.search ? d.query.search : []).map(s => s.title);
}

async function wikiFetch(title) {
  const url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=' +
    encodeURIComponent(title);
  const r = await fetch(url, { headers: { 'User-Agent': 'VectorDB-RAG/1.0' } });
  const d = await r.json();
  const pages = d.query && d.query.pages ? Object.values(d.query.pages) : [];
  for (const pg of pages) {
    if (pg.extract && pg.extract.trim()) return { title: pg.title, text: pg.extract };
  }
  return null;
}

async function storeChunk(chunkTitle, text, emb) {
  const p = await getPool();
  if (p) {
    const r = await p.query('INSERT INTO doc_chunks (title, text, embedding) VALUES ($1, $2, $3::vector) RETURNING id', [chunkTitle, text, vecToString(emb)]);
    return r.rows[0].id;
  }
  return docDB.insert(chunkTitle, text, emb);
}

// Keyword buckets mirror index.html so a web-fetched article lands in the same
// region of the 16D graph as a manually inserted document about that topic.
const KW = {
  cs:     ['algorithm','data','tree','graph','array','linked','hash','stack','queue','sort','binary','dynamic','programming','recursion','complexity','pointer','node','search','insert','bfs','dfs','heap','trie','database','index','query','sql','vector','embedding','semantic','network','distributed','cache','memory','runtime','thread','process','async','library','framework'],
  math:   ['calculus','matrix','probability','theorem','integral','derivative','linear','algebra','equation','function','prime','modular','combinatorics','permutation','eigenvalue','statistics','proof','geometry','trig','logarithm','limit','sequence','series','fraction','ratio'],
  food:   ['food','pizza','sushi','ramen','pasta','recipe','cook','eat','restaurant','dish','ingredient','flavor','spice','noodle','bread','croissant','taco','fish','rice','soup','biryani','curry','kebab','tikka','samosa','dosa','naan','paneer','masala','burger','fries','steak','chicken','beef','pork','lamb','cheese','egg','meat','grill','barbecue','roast','fried','salad','sandwich','pancake','waffle','cake','cookie','pie','pastry','chocolate','dessert','cream','coffee','tea','wine','beer','juice','tomato','onion','garlic','mango','banana'],
  sports: ['sport','basketball','football','tennis','chess','swim','game','play','score','team','athlete','competition','match','tournament','olympic','dribble','tackle','serve','soccer','cricket','hockey','golf','boxing','wrestling','cycling','running','marathon','yoga','gym','fitness','baseball','volleyball','rugby','badminton']
};

// Deterministic 16D embedding for a graph point (frontend adds ±0.02 jitter).
function graphEmbedding(text) {
  const t = text.toLowerCase(), ws = t.split(/\s+/);
  const s = { cs: 0, math: 0, food: 0, sports: 0 };
  for (const w of ws)
    for (const [cat, kws] of Object.entries(KW))
      for (const kw of kws) if (w.includes(kw) || kw.startsWith(w)) { s[cat] += 0.35; break; }
  const mx = Math.max(...Object.values(s), 0.01);
  const n = v => Math.min(v / mx * 0.88, 0.94);
  const emb = new Array(16).fill(0.08);
  const fill = (i, score) => {
    if (score < 0.01) return;
    const b = n(score);
    emb[i] = Math.max(0.05, b); emb[i + 1] = Math.max(0.05, b);
    emb[i + 2] = Math.max(0.05, b * 0.92); emb[i + 3] = Math.max(0.05, b * 0.87);
  };
  fill(0, s.cs); fill(4, s.math); fill(8, s.food); fill(12, s.sports);
  return emb;
}

async function graphPointExists(meta) {
  const p = await getPool();
  if (p) {
    const r = await p.query("SELECT 1 FROM items WHERE metadata = $1 AND category = 'doc' LIMIT 1", [meta]);
    return r.rows.length > 0;
  }
  return vdb.all().some(i => i.metadata === meta && i.category === 'doc');
}

// Pulls the top article for a topic, chunks it, and embeds up to 10 chunks.
// Also used by the "Search the web for more" button in the Ask AI tab.
async function webIngest(topic, maxArticles = 1) {
  await ensureDemo();
  const titles = await wikiSearch(topic);
  if (!titles.length) return { error: 'No Wikipedia results for "' + topic + '".' };
  const added = [];
  for (const title of titles.slice(0, maxArticles)) {
    const art = await wikiFetch(title);
    if (!art) continue;
    const chunks = chunkText(art.text, 250, 30);
    const cap = chunks.slice(0, 10);
    const embs = await Promise.all(cap.map(c => providers.embedOne(c)));
    const ids = [];
    for (let i = 0; i < cap.length; i++) {
      const emb = embs[i];
      if (!emb) continue;
      const chunkTitle = cap.length > 1 ? art.title + ' [' + (i + 1) + '/' + cap.length + ']' : art.title;
      ids.push(await storeChunk(chunkTitle, cap[i], emb));
    }
    // Give the article a 16D point on the visualizer graph so fetched
    // knowledge shows up next to manually inserted documents.
    if (!(await graphPointExists(art.title))) {
      await insertItem(art.title, 'doc', graphEmbedding(art.title));
    }
    added.push({ title: art.title, chunks: cap.length, stored: ids.length, truncated: chunks.length > cap.length });
  }
  if (!added.length) return { error: 'Could not fetch article content.' };
  const dims = (await getPool()) ? 1536 : docDB.getDims();
  const total = added.reduce((s, a) => s + a.stored, 0);
  return { added, dims, message: 'Stored ' + total + ' chunk(s) from the web.' };
}

// status / stats

async function status() {
  await ensureDemo();
  const p = await getPool();
  let docCount, docDims;
  if (p) {
    const r = await p.query('SELECT COUNT(*)::int AS n FROM doc_chunks');
    docCount = r.rows[0].n;
    docDims = docCount > 0 ? 1536 : 0;
  } else {
    docCount = docDB.size();
    docDims = docDB.getDims();
  }
  const [embUp, groqUp] = await Promise.all([providers.embedAvailable(), providers.groqAvailable()]);
  return {
    groqAvailable: groqUp,
    apiKeySet: !!process.env.GROQ_API_KEY,
    embedAvailable: embUp,
    embedKeySet: !!process.env.OPENROUTER_API_KEY,
    embedModel: providers.EMBED_MODEL,
    genModel: providers.GEN_MODEL,
    docCount,
    docDims,
    demoDims: DIMS,
    demoCount: vdb.size()
  };
}

function stats() {
  return { count: vdb.size(), dims: DIMS, algorithms: ['bruteforce', 'kdtree', 'hnsw'], metrics: ['euclidean', 'cosine', 'manhattan'] };
}

module.exports = {
  items, searchItems, insertItem, deleteItem, benchmark, hnswInfo,
  docInsert, docList, docDelete, docSearch, docAsk, webIngest,
  status, stats
};
