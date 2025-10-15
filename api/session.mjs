// /api/session.mjs
export const config = { runtime: "nodejs" };
const SESSIONS = globalThis.__HCA_SESSIONS__ ?? (globalThis.__HCA_SESSIONS__ = new Map());

export default async function handler(req) {
  if (req.method !== "GET") {
    return new Response("Method not allowed", { status: 405 });
  }
  const sid = req.headers.get("x-session-id") || "anon";
  const history = SESSIONS.get(sid)?.history ?? [];
  return new Response(JSON.stringify({ sessionId: sid, messages: history }), {
    headers: { "content-type": "application/json" }
  });
}
