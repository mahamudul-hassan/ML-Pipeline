// Vercel Edge Function: lists the models available to the user's Ollama Cloud key.
export const config = { runtime: 'edge' };

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });

export default async function handler(req) {
  const env = globalThis.process?.env || {};
  const key = (req.headers.get('x-ollama-key') || (env.ALLOW_SERVER_KEY === 'true' ? env.OLLAMA_API_KEY : '') || '').trim();
  try {
    const res = await fetch('https://ollama.com/api/tags', { headers: key ? { Authorization: `Bearer ${key}` } : {} });
    const text = await res.text();
    return new Response(text, { status: res.status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });
  } catch (e) {
    return json({ error: `Could not reach Ollama Cloud: ${e.message}` }, 502);
  }
}
