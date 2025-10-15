// /api/reset.mjs
export const config = { runtime: "nodejs" };
const SESSIONS = globalThis.__HCA_SESSIONS__ ?? (globalThis.__HCA_SESSIONS__ = new Map());

export default async function handler(req) {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }
  const cookie = req.headers.get("cookie") || "";
  const sidFromCookie = cookie.match(/(?:^|;\\s*)hca_sid=([^;]+)/)?.[1];
  let body = {};
  try { body = await req.json(); } catch {}
  const sid = body.sessionId || req.headers.get("x-session-id") || sidFromCookie || "anon";
  SESSIONS.delete(sid);
  return new Response(JSON.stringify({ ok: true, sessionId: sid }), {
    headers: { "content-type": "application/json" }
  });
}
