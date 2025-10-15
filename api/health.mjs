// /api/health.mjs
export const config = { runtime: "nodejs" };

export default async function handler() {
  return new Response(JSON.stringify({ ok: true, service: "hca-chatbot", ts: Date.now() }), {
    headers: { "content-type": "application/json", "cache-control": "no-store" }
  });
}
