// Vercel Edge Function: streams Ollama Cloud /api/chat responses to the browser.
// The user's key arrives in the x-ollama-key header and is only forwarded to ollama.com.
export const config = { runtime: 'edge' };

const UPSTREAM = 'https://ollama.com';
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });

function pickKey(req) {
  const k = req.headers.get('x-ollama-key');
  if (k) return k.trim();
  const env = globalThis.process?.env || {};
  return env.ALLOW_SERVER_KEY === 'true' && env.OLLAMA_API_KEY ? env.OLLAMA_API_KEY : '';
}

export default async function handler(req) {
  if (req.method !== 'POST') return json({ error: 'Use POST.' }, 405);
  const key = pickKey(req);
  if (!key) return json({ error: 'Missing Ollama API key. Add it in Settings → AI.' }, 401);
  const body = await req.text();
  if (body.length > 4_000_000) return json({ error: 'Request too large.' }, 413);
  let upstream;
  try {
    upstream = await fetch(`${UPSTREAM}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body });
  } catch (e) {
    return json({ error: `Could not reach Ollama Cloud: ${e.message}` }, 502);
  }
  return new Response(upstream.body, { status: upstream.status, headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/x-ndjson', 'Cache-Control': 'no-store' } });
}
