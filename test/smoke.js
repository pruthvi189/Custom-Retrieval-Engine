// Local smoke test — run `vercel dev` first (defaults to port 3000), then `npm test`.

const BASE = 'http://localhost:3000/api';

async function test() {
  const status = await (await fetch(BASE + '/status')).json();
  console.log('status:', { hasGroq: status.apiKeySet, hasEmbed: status.embedKeySet });

  const stats = await (await fetch(BASE + '/stats')).json();
  console.log('stats:', stats.count + ' vectors');

  const demoEmb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
  const ins = await fetch(BASE + '/insert', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ metadata: 'smoke vector', category: 'test', embedding: demoEmb })
  });
  const { id } = await ins.json();
  console.log('inserted id:', id);

  const qStr = demoEmb.map(v => v.toFixed(4)).join(',');
  const searchUrl = BASE + '/search?v=' + qStr + '&k=1&algo=bruteforce';
  const s = await fetch(searchUrl);
  const data = await s.json();
  console.log('search:', data.results.length + ' results');

  const del = await fetch(BASE + '/delete/' + id, { method: 'DELETE' });
  const d = await del.json();
  console.log('delete ok:', d.ok);

  console.log('Smoke test passed');
}

if (require.main === module) test().catch(console.error);
