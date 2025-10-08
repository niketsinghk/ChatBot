// server.mjs
import "dotenv/config";
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ===================== Config ===================== */
const PORT = parseInt(process.env.PORT || "5173", 10);
const DATA_DIR = path.join(__dirname, "data");
const EMB_PATH = path.join(DATA_DIR, "index.json"); // your chosen filename
const TOP_K = parseInt(process.env.TOP_K || "6", 10);

// Use supported defaults; override via .env if you like `gemini-1.5-flash`
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";

/* ===================== App ===================== */
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.use("/images", express.static(path.join(__dirname, "images")));

if (!process.env.GOOGLE_API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm = genAI.getGenerativeModel({ model: GENERATION_MODEL });

/* ===================== Hinglish/English Detection ===================== */
function detectResponseMode(q) {
  const text = (q || "").toLowerCase();

  // Devanagari → treat as Hinglish output (Latin), not Devanagari
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  if (hasDevanagari) return "hinglish";

  const hinglishTokens = [
    "hai","hain","tha","thi","the","kya","kyu","kyun","kyunki","kisi","kis",
    "kaun","kab","kaha","kahaan","kaise","nahi","nahin","ka","ki","ke","mein","me","mai","mei",
    "hum","ap","aap","tum","kr","kar","karo","karna","chahiye","bhi","sirf","jaldi","kitna",
    "kab","kaha","kaise","ho","hoga","hogaya","krdo","pls","plz","yaar"
  ];

  let score = 0;
  for (const t of hinglishTokens) {
    if (text.includes(` ${t} `) || text.startsWith(t + " ") || text.endsWith(" " + t) || text === t) score += 1;
  }
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  score += chatCues >= 1 ? 0.5 : 0;

  return score >= 2 ? "hinglish" : "english";
}

/* ===================== Stopwords + Cleaner ===================== */
const EN_STOPWORDS = new Set(`a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's
me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves`.trim().split(/\s+/));

const HINGLISH_STOPWORDS = new Set([
  "hai","hain","ho","hona","hoga","hogi","honge","hote","hota","thi","tha","the",
  "kya","kyu","kyun","kyunki","kisi","kis","kaun","konsa","kab","kaha","kahaan","kaise",
  "nahi","nahin","na","mat","bas","sirf","bhi","hi","to","tho","ab","abhi","phir","fir",
  "ye","yeh","vo","woh","aisa","waisa","jab","tab","agar","lekin","magar","par","per","ya","aur",
  "ka","ki","ke","mein","me","mai","mei","mujhe","mujhko","hume","humko","tumhe","aap","ap","hum","tum",
  "se","ko","tak","pe","par","liye","ke","liye",
  "kr","kar","karo","karna","karke","krke","krna","ho gya","hogaya","chahiye","chahie","krdo","kardo","de","do","lo","le","dena","lena"
]);

const HINDI_STOPWORDS = new Set([
  "है","हैं","हो","होना","होगा","होगी","होंगे","होते","होता","था","थी","थे",
  "क्या","क्यों","क्योंकि","किसी","कौन","कौनसा","कब","कहाँ","कैसे",
  "नहीं","मत","बस","सिर्फ","भी","ही","तो","अब","अभी","फिर",
  "यह","ये","वह","वो","जब","तब","अगर","लेकिन","मगर","या","और",
  "का","की","के","में","मे","मुझे","हमें","तुम्हें","आप","हम","तुम",
  "से","को","तक","पर","लिए","चाहिए","कर","करो","करना","करके","कर दें","कर लो"
]);

const PROTECTED_TOKENS = new Set([
  "hca","hari","chand","anand","anil","anand","duke","kansai","special","highlead","merrow","megasew","amf","reece","delhi","india","solution","solutions","automation","garment","leather","mattress"
]);

function cleanForEmbedding(s) {
  if (!s) return "";
  const lower = s.toLowerCase();
  const stripped = lower.replace(/[^a-z0-9\u0900-\u097F\s]/g, " ");
  const tokens = stripped.split(/\s+/).filter(Boolean);
  const kept = tokens.filter(t => {
    if (PROTECTED_TOKENS.has(t)) return true;
    if (EN_STOPWORDS.has(t)) return false;
    if (HINGLISH_STOPWORDS.has(t)) return false;
    if (HINDI_STOPWORDS.has(t)) return false;
    return true;
  });
  return kept.join(" ").trim();
}

/* ===================== Vectors & Utils ===================== */
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function loadVectors() {
  if (!fs.existsSync(EMB_PATH)) {
    throw new Error(`Embeddings not found at ${EMB_PATH}. Run "npm run embed" first.`);
  }
  const raw = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
  if (!raw?.vectors?.length) throw new Error("Embeddings file has no vectors.");
  return raw.vectors;
}

let VECTORS = [];
try {
  VECTORS = loadVectors();
  console.log(`🗂️  Loaded ${VECTORS.length} vectors (stopwords: EN+Hinglish+Hindi)`);
} catch (err) {
  console.warn("⚠️", err.message);
}

/* ===================== Routes ===================== */
app.get("/api/health", (_, res) => res.json({ ok: true }));

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body || {};
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Missing 'question' string" });
    }

    // --- Polite small-talk (handled here, no PDF needed) ---
    const q = question.trim();
    const smallTalkHandlers = [
      {
        test: /^(hi|hello|hey|hlo|namaste|hola|yo)\b/i,
        reply: "Hello! 👋 I’m HCA’s assistant. Ask me anything about HCA—for example: “Which brands do we represent?” or “What’s our mission?”"
      },
      {
        test: /^(good\s*(morning|afternoon|evening|night))\b/i,
        reply: "Hello! 👋 Hope you’re having a great day. Ask me anything about HCA—happy to help."
      },
      {
        test: /^(thanks|thank you|thx|ty)\b/i,
        reply: "You’re welcome! If you need anything else about HCA, just ask. 🙂"
      },
      {
        test: /(who are you|what can you do|help|menu)/i,
        reply: "I’m HCA’s PDF-grounded assistant. I answer strictly from our knowledge base. Try: “About HCA”, “Which industries do we serve?”, or “What does it say about Duke?”"
      },
      { test: /^about\s*hca\b/i, reply: null } // let retrieval handle
    ];
    for (const h of smallTalkHandlers) {
      if (h.test.test(q) && h.reply) {
        return res.json({ answer: h.reply, citations: [], mode: detectResponseMode(q) });
      }
    }

    if (!VECTORS.length) {
      return res.status(500).json({ error: "Embeddings not loaded. Run `npm run embed` first." });
    }

    const mode = detectResponseMode(q);

    // Cleaned query for embedding (for better recall)
    const cleanedQuery = cleanForEmbedding(q) || q.toLowerCase();

    // ---- Embed query (support both shapes returned by SDK) ----
    // ---- Embed query (text-embedding-004 expects `content`, not `contents`) ----
const embRes = await embedder.embedContent({
  content: { parts: [{ text: cleanedQuery }] }
});
const qVec =
  embRes?.embedding?.values ||  // correct for text-embedding-004
  embRes?.embeddings?.[0]?.values || // extra safeguard
  [];


    // ---- Retrieve ----
    const scored = VECTORS
      .map(v => ({ ...v, score: cosineSim(qVec, v.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    const contextBlocks = scored
      .map((s, i) => `【${i+1}】 ${s.text_original || s.text_cleaned || s.text}`)
      .join("\n\n");

    const languageGuide =
      mode === "hinglish"
        ? `REPLY LANGUAGE: Hinglish (Hindi in Latin script, e.g., "HCA ka focus automation par hai"). Do NOT use Devanagari.`
        : `REPLY LANGUAGE: English. Professional and concise.`;

    const systemInstruction = `
You are HCA's internal assistant. Answer STRICTLY and ONLY from the provided CONTEXT (the HCA knowledge.pdf).
If the answer is not present in the CONTEXT, reply exactly:
"I don't have this information in the provided HCA knowledge base."

Rules:
- Do not invent or add external knowledge.
- Be concise and factual.
- ${languageGuide}
`.trim();

    const prompt = `
${systemInstruction}

QUESTION:
${q}

CONTEXT (numbered blocks):
${contextBlocks}

Format:
- Direct answer grounded in context.
- If not found: "I don't have this information in the provided HCA knowledge base."
- Use the reply language specified above.
`.trim();

    const result = await llm.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }]}]
    });

    const text = result.response.text();
    return res.json({
      answer: text,
      mode,
      citations: scored.map((s, i) => ({ idx: i+1, score: s.score }))
    });

  } catch (err) {
    console.error("Ask error:", err);
    const status = err?.status || 500;
    const msg = err?.message || err?.statusText || "Generation failed";
    return res.status(status).json({
      error: msg,
      details: { status, statusText: err?.statusText || null, type: err?.name || null }
    });
  }
});

/* ===================== Start ===================== */
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📎 Static UI at http://localhost:${PORT}/`);
});
