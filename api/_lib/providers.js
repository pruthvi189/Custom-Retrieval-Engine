// Provider clients — OpenRouter for embeddings, Groq for generation.
// Keys come from env vars only, never from source files.

const EMBED_MODEL = process.env.OPENROUTER_EMBED_MODEL || 'openai/text-embedding-3-small';
const GEN_MODEL = process.env.GROQ_GEN_MODEL || 'llama-3.3-70b-versatile';
const embedKey = process.env.OPENROUTER_API_KEY || '';
const groqKey = process.env.GROQ_API_KEY || '';

async function embedOne(text) {
  if (!embedKey) return null;
  const r = await fetch('https://openrouter.ai/api/v1/embeddings', {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + embedKey, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, input: text })
  });
  if (!r.ok) return null;
  const d = await r.json();
  const e = d && d.data && d.data[0] && d.data[0].embedding;
  return Array.isArray(e) ? e : null;
}

async function embedAvailable() {
  if (!embedKey) return false;
  try {
    const e = await embedOne('test');
    return Array.isArray(e) && e.length > 0;
  } catch (_) { return false; }
}

async function generate(prompt) {
  if (!groqKey) return 'ERROR: GROQ_API_KEY is not set. Add it as a Vercel env var and redeploy.';
  try {
    const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + groqKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: GEN_MODEL, messages: [{ role: 'user', content: prompt }], temperature: 0.7, stream: false })
    });
    if (!r.ok) {
      let msg = 'Groq API request failed (HTTP ' + r.status + ')';
      try {
        const d = await r.json();
        if (d && d.error && d.error.message) msg = d.error.message;
      } catch (_) {}
      return 'ERROR: ' + msg;
    }
    const d = await r.json();
    const ans = d && d.choices && d.choices[0] && d.choices[0].message && d.choices[0].message.content;
    return ans ? ans : 'ERROR: Empty response from Groq';
  } catch (e) {
    return 'ERROR: Could not reach the Groq API. Check your internet connection.';
  }
}

async function groqAvailable() {
  if (!groqKey) return false;
  try {
    const r = await fetch('https://api.groq.com/openai/v1/models', {
      headers: { Authorization: 'Bearer ' + groqKey }
    });
    return r.ok;
  } catch (_) { return false; }
}

module.exports = { EMBED_MODEL, GEN_MODEL, embedOne, embedAvailable, generate, groqAvailable };
