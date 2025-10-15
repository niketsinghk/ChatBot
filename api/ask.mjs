// Vercel serverless version of your /api/ask route (no Express needed)
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

// -------- Paths & setup (works on Vercel) --------
const DATA_DIR = path.join(process.cwd(), "data");
const EMB_PATH = path.join(DATA_DIR, "index.json");
const TOP_K = parseInt(process.env.TOP_K || "6", 10);
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";

if (!process.env.GOOGLE_API_KEY) {
  throw new Error("Missing GOOGLE_API_KEY env on Vercel");
}

const genAI   = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder= genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm     = genAI.getGenerativeModel({ model: GENERATION_MODEL });

// -------- Helpers copied from your server.mjs (trimmed where possible) --------
function detectResponseMode(q=""){
  const text = q.toLowerCase();
  if (/[\u0900-\u097F]/.test(text)) return "hinglish";
  const hits = ["hai","hain","kya","kyu","kyunki","kab","kaha","kaise","nahi","ka","ki","ke","mein","me","mai","hum","aap","tum","kr","kar","karo","chahiye","bhi"]
    .reduce((n,t)=> n + (text.includes(` ${t} `)||text.startsWith(t+" ")||text.endsWith(" "+t)||text===t), 0);
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  return (hits + (chatCues?0.5:0)) >= 2 ? "hinglish" : "english";
}

const EN_STOP = new Set("a about above after again against all am an and any are as at be because been before being below between both but by do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself me more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with would you your yours yourself yourselves".split(/\s+/));
const PROTECT = new Set(["hca","hari","chand","anand","duke","vios","merrow","delhi","india","automation","garment","leather","solutions"]);

function cleanForEmbedding(s=""){
  const lower = s.toLowerCase();
  const stripped = lower.replace(/[^a-z0-9\u0900-\u097F\s]/g," ");
  return stripped.split(/\s+/).filter(Boolean).filter(t=>{
    if (PROTECT.has(t)) return true;
    if (EN_STOP.has(t)) return false;
    return true;
  }).join(" ");
}

function cosineSim(a,b){
  let dot=0, na=0, nb=0, n=Math.min(a.length,b.length);
  for(let i=0;i<n;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) || 1);
}

function loadVectors(){
  if (!fs.existsSync(EMB_PATH)) throw new Error(`Embeddings not found at ${EMB_PATH}.`);
  const raw = JSON.parse(fs.readFileSync(EMB_PATH,"utf8"));
  if (!raw?.vectors?.length) throw new Error("Embeddings file has no vectors.");
  return raw.vectors;
}

// Load once per cold start
let VECTORS = [];
try { VECTORS = loadVectors(); } catch(e){ console.warn(e.message); }

// -------- Serverless handler --------
export default async function handler(req, res){
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  try {
    // Vercel may hand you an object already; accept both message|question
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const q = (body.message ?? body.question ?? "").trim();
    if (!q) return res.status(400).json({ error: "Missing 'message' or 'question'." });

    // Small-talk shortcuts like your server (kept minimal)
    if (/^(hi|hello|hey|namaste)\b/i.test(q)) {
      return res.json({ answer: "Hello! 👋 I’m HCA’s assistant. Ask me anything about HCA.", citations: [], mode: detectResponseMode(q) });
    }

    if (!VECTORS.length) {
      return res.status(500).json({ error: "Embeddings not loaded on server. Add data/index.json and redeploy." });
    }

    const mode = detectResponseMode(q);
    const cleaned = cleanForEmbedding(q) || q.toLowerCase();

    // Embed query (text-embedding-004 uses `content.parts[].text`)
    const embRes = await embedder.embedContent({ content: { parts: [{ text: cleaned }] } });
    const qVec = embRes?.embedding?.values || embRes?.embeddings?.[0]?.values || [];
    if (!qVec.length) return res.status(500).json({ error: "Embedding failed" });

    // Retrieve
    const top = VECTORS
      .map(v => ({ ...v, score: cosineSim(qVec, v.embedding) }))
      .sort((a,b)=> b.score - a.score)
      .slice(0, TOP_K);

    const context = top.map((s,i)=> `【${i+1}】 ${s.text_original || s.text_cleaned || s.text}`).join("\n\n");

    const languageGuide = mode === "hinglish"
      ? `REPLY LANGUAGE: Hinglish (Hindi in Latin script).`
      : `REPLY LANGUAGE: English. Professional and concise.`;

    const systemInstruction = `
Answer STRICTLY and ONLY from the provided CONTEXT (HCA knowledge).
If not present in CONTEXT, reply exactly:
"I don't have this information in the provided HCA knowledge base."
- No external facts. Be concise.
- ${languageGuide}
`.trim();

    const prompt = `
${systemInstruction}

QUESTION:
${q}

CONTEXT:
${context}
`.trim();

    const result = await llm.generateContent({ contents:[{ role:"user", parts:[{ text: prompt }]}] });
    const text = result?.response?.text?.() || "I don't have this information in the provided HCA knowledge base.";

    return res.status(200).json({
      answer: text,
      mode,
      citations: top.map((s,i)=> ({ idx:i+1, score:s.score }))
    });
  } catch (err) {
    console.error("ask error:", err);
    return res.status(err?.status || 500).json({
      error: err?.message || "Server error",
      details: { status: err?.status || 500, statusText: err?.statusText || null, type: err?.name || null }
    });
  }
}