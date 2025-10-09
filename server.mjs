// server.mjs
import "dotenv/config";
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import cookieParser from "cookie-parser";
import { v4 as uuidv4 } from "uuid";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ===================== Config ===================== */
const PORT = parseInt(process.env.PORT || "5173", 10);
const DATA_DIR = path.join(__dirname, "data");
const EMB_PATH = path.join(DATA_DIR, "index.json");
const TOP_K = parseInt(process.env.TOP_K || "6", 10);

const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";

if (!process.env.GOOGLE_API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY in .env");
  process.exit(1);
}

/* ===================== App ===================== */
const app = express();
app.use(
  cors({
    origin: true,          // set to your frontend origin(s) in prod
    credentials: true,     // allow cookie
  })
);
app.use(express.json({ limit: "2mb" }));
app.use(cookieParser());

// Static UI
app.use(express.static(path.join(__dirname, "public")));
app.use("/images", express.static(path.join(__dirname, "images")));

/* ===================== Gemini ===================== */
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm = genAI.getGenerativeModel({ model: GENERATION_MODEL });

/* ===================== Session Store ===================== */
const sessions = new Map(); // sid -> { history: [], createdAt, lastSeen }

function getTimeOfDayGreeting() {
  const h = new Date().getHours();
  if (h < 5) return "Good night";
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  if (h < 21) return "Good evening";
  return "Hello";
}

function sessionMiddleware(req, res, next) {
  let sid = req.get("X-Session-ID") || req.body?.sessionId || req.cookies?.sid;
  if (!sid || typeof sid !== "string" || sid.length > 200) {
    sid = uuidv4();
    res.cookie("sid", sid, {
      httpOnly: true,
      sameSite: "Lax",
      secure: !!process.env.COOKIE_SECURE, // set COOKIE_SECURE=1 behind HTTPS
      maxAge: 1000 * 60 * 60 * 24 * 30,    // 30 days
    });
  }
  const now = Date.now();
  if (!sessions.has(sid)) sessions.set(sid, { history: [], createdAt: now, lastSeen: now });
  else sessions.get(sid).lastSeen = now;

  req.sid = sid;
  req.session = sessions.get(sid);
  next();
}

/* ===================== Language Mode ===================== */
function detectResponseMode(q) {
  const text = (q || "").toLowerCase();
  if (/[\u0900-\u097F]/.test(text)) return "hinglish";

  const hinglishTokens = [
    "hai","hain","tha","thi","the","kya","kyu","kyun","kyunki","kisi","kis",
    "kaun","kab","kaha","kahaan","kaise","nahi","nahin","ka","ki","ke","mein","me","mai","mei",
    "hum","ap","aap","tum","kr","kar","karo","karna","chahiye","bhi","sirf","jaldi","kitna",
    "ho","hoga","hogaya","krdo","pls","plz","yaar","shukriya","dhanyavaad","dhanyavad"
  ];
  let score = 0;
  for (const t of hinglishTokens) {
    if (text.includes(` ${t} `) || text.startsWith(t + " ") || text.endsWith(" " + t) || text === t) score += 1;
  }
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  score += chatCues >= 1 ? 0.5 : 0;
  return score >= 2 ? "hinglish" : "english";
}

/* ===================== Stopwords & Cleaner ===================== */
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
  const kept = tokens.filter((t) => {
    if (PROTECTED_TOKENS.has(t)) return true;
    if (EN_STOPWORDS.has(t)) return false;
    if (HINGLISH_STOPWORDS.has(t)) return false;
    if (HINDI_STOPWORDS.has(t)) return false;
    return true;
  });
  return kept.join(" ").trim();
}

/* ===================== Vectors ===================== */
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
function loadVectors() {
  if (!fs.existsSync(EMB_PATH)) throw new Error(`Embeddings not found at ${EMB_PATH}. Run "npm run embed" first.`);
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

/* ===================== Small Talk ===================== */
function makeSmallTalkReply(kind, mode) {
  const en = {
    hello: [
      `${getTimeOfDayGreeting()}! 👋 I’m HCA’s assistant. Ask me anything about HCA (brands, machines, spares, etc.).`,
      `Hello! 👋 How can I help you with HCA today?`,
      `Hi! 👋 I’m here for HCA queries—try “About HCA”, “Duke-Jia details”, or “Spares info”.`,
    ],
    morning: [`Good morning! ☀️ How can I help with HCA today?`],
    afternoon: [`Good afternoon! 😊 What would you like to know about HCA?`],
    evening: [`Good evening! 🌙 Need help with HCA machines or spares?`],
    thanks: [
      `You’re welcome! 🙏 Anything else I can do for you about HCA?`,
      `Happy to help! If you need more info, just ask. 🙂`,
    ],
    bye: [
      `Take care! 👋 If you need HCA help later, I’m here.`,
      `Bye! Have a great day. 👋`,
    ],
    help: [
      `I answer from HCA’s knowledge base. You can ask: “Which brands do we represent?”, “About Duke-Jia E+P flagship”, or “Industries we serve?”`,
    ],
    ack: [
      `Got it! 👍 What would you like to ask about HCA next?`,
      `Okay. Tell me your HCA question—brands, machines, or spares.`,
    ],
  };

  const hi = {
    hello: [
      `Namaste! 👋 HCA assistant bol raha hoon. HCA se related kuch bhi puchhiye (brands, machines, spares).`,
      `Hello ji! 👋 HCA ke baare mein madad chahiye?`,
      `Hi! 👋 Aap HCA queries puchh sakte ho—“About HCA”, “Duke-Jia details”, “Spares info”.`,
    ],
    morning: [`Good morning! ☀️ Aaj HCA mein kis cheez mein help chahiye?`],
    afternoon: [`Good afternoon! 😊 HCA ke baare mein kya jaan’na chahoge?`],
    evening: [`Good evening! 🌙 HCA machines/spares par madad chahiye to batayein.`],
    thanks: [
      `Shukriya! 🙏 Aur kuch madad chahiye to pooch lijiye.`,
      `Welcome ji! 🙂 Aur koi HCA info chahiye?`,
    ],
    bye: [
      `Theek hai, milte hain! 👋 Jab chahein HCA help ke liye ping kar dijiyega.`,
      `Bye! 👋 Din shubh rahe.`,
    ],
    help: [
      `Main HCA ke knowledge base se answer karta hoon. Aap pooch sakte ho: “Hum kin brands ko represent karte hain?”, “Duke-Jia E+P flagship kya hai?”, “Hum kaun-kaun si industries serve karte hain?”`,
    ],
    ack: [
      `Thik hai! 👍 Ab HCA ke baare mein kya puchhna hai?`,
      `Okay ji. HCA—brands, machines ya spares—kis par info chahiye?`,
    ],
  };

  const bank = mode === "hinglish" ? hi : en;
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  switch (kind) {
    case "hello": return pick(bank.hello);
    case "morning": return pick(bank.morning);
    case "afternoon": return pick(bank.afternoon);
    case "evening": return pick(bank.evening);
    case "thanks": return pick(bank.thanks);
    case "bye": return pick(bank.bye);
    case "help": return pick(bank.help);
    case "ack": return pick(bank.ack);
    default: return pick(bank.hello);
  }
}

function smallTalkMatch(q) {
  const t = (q || "").trim();
  const patterns = [
    { kind: "hello",     re: /^(hi+|h[iy]+|hello+|hey( there)?|hlo+|hloo+|yo+|hola|namaste|namaskar|salaam|salam|sup|wass?up|what'?s up|👋|🙏)\b/i },
    { kind: "morning",   re: /^good\s*morning\b/i },
    { kind: "afternoon", re: /^good\s*afternoon\b/i },
    { kind: "evening",   re: /^good\s*evening\b/i },
    { kind: "ack",       re: /^(ok+|okay+|okk+|hmm+|haan+|ha+|sure|done|great|nice|cool|perfect|thik|theek|fine)\b/i },
    { kind: "thanks",    re: /^(thanks|thank\s*you|thx|ty|much\s*(appreciated|thanks)|many\s*thanks|appreciate(d)?|shukriya|dhanyavaad|dhanyavad)\b/i },
    { kind: "bye",       re: /^(bye|good\s*bye|goodbye|see\s*ya|see\s*you|take\s*care|tc|catch\s*you\s*later)\b/i },
    { kind: "help",      re: /(who\s*are\s*you|what\s*can\s*you\s*do|help|menu|options|how\s*to\s*use)\b/i },
  ];
  for (const p of patterns) if (p.re.test(t)) return p.kind;
  return null;
}

/* ===================== Health & Utilities ===================== */
app.get("/api/health", (_, res) => res.json({ ok: true, ts: Date.now() }));

app.post("/api/reset", sessionMiddleware, (req, res) => {
  req.session.history = [];
  res.json({ sessionId: req.sid, cleared: true });
});

app.get("/api/session", sessionMiddleware, (req, res) => {
  res.json({
    sessionId: req.sid,
    historyLength: req.session.history.length,
    createdAt: req.session.createdAt,
    lastSeen: req.session.lastSeen,
  });
});

/* ===================== Ask ===================== */
app.post("/api/ask", sessionMiddleware, async (req, res) => {
  try {
    const question = (req.body?.question ?? req.body?.message ?? "").toString();
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Missing 'question' (or 'message') string" });
    }

    const q = question.trim();
    const mode = detectResponseMode(q);

    // 1) Ultra-short guard → treat as greeting
    const lettersOnly = q.replace(/[\W_]+/g, "");
    if (lettersOnly.length <= 3) {
      const reply = makeSmallTalkReply("hello", mode);
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: reply, ts: Date.now() });
      return res.json({ answer: reply, sessionId: req.sid, mode, citations: [] });
    }

    // 2) Small talk detector
    const kind = smallTalkMatch(q);
    if (kind) {
      const reply = makeSmallTalkReply(kind, mode);
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: reply, ts: Date.now() });
      return res.json({ answer: reply, sessionId: req.sid, mode, citations: [] });
    }

    // 3) RAG route
    if (!VECTORS.length) {
      const fallback = mode === "hinglish"
        ? "Embeddings load nahi hue. Pehle `npm run embed` chalaa kar knowledge base taiyaar kijiye."
        : "Embeddings are not loaded. Please run `npm run embed` to prepare the knowledge base.";
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: fallback, ts: Date.now() });
      return res.status(500).json({ answer: fallback, sessionId: req.sid, mode, citations: [] });
    }

    const cleanedQuery = cleanForEmbedding(q) || q.toLowerCase();

    // Embed query (text-embedding-004)
    const embRes = await embedder.embedContent({
      content: { parts: [{ text: cleanedQuery }] },
    });
    const qVec = embRes?.embedding?.values || embRes?.embeddings?.[0]?.values || [];

    // Retrieve
    const scored = VECTORS
      .map((v) => ({ ...v, score: cosineSim(qVec, v.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    // Optional: soft fallback when nothing matches well
    const MIN_OK_SCORE = 0.18;
    if (scored.length === 0 || (scored?.[0]?.score ?? 0) < MIN_OK_SCORE) {
      const tip = mode === "hinglish"
        ? "Is topic par knowledge base mein clear info nahi mil rahi. Thoda specific likhiye—jaise 'Duke-Jia E+P flagship features' ya 'Highlead 269 application'."
        : "I couldn’t find clear context in the knowledge base for that. Try being more specific—for example, 'Duke-Jia E+P flagship features' or 'Highlead 269 application'.";
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: tip, ts: Date.now() });
      return res.json({ answer: tip, sessionId: req.sid, mode, citations: [] });
    }

    const contextBlocks = scored
      .map((s, i) => `【${i + 1}】 ${s.text_original || s.text_cleaned || s.text}`)
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

    // Append user turn
    req.session.history.push({ role: "user", content: q, ts: Date.now() });

    const result = await llm.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });
    const text = result.response.text();

    // Append assistant turn
    req.session.history.push({ role: "assistant", content: text, ts: Date.now() });

    res.json({
      answer: text,
      mode,
      sessionId: req.sid,
      citations: scored.map((s, i) => ({ idx: i + 1, score: s.score })),
    });
  } catch (err) {
    console.error("Ask error:", err);
    const status = err?.status || 500;
    const msg = err?.message || err?.statusText || "Generation failed";
    res.status(status).json({
      error: msg,
      details: { status, statusText: err?.statusText || null, type: err?.name || null },
    });
  }
});

/* ===================== Start ===================== */
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📎 Static UI at http://localhost:${PORT}/`);
});
