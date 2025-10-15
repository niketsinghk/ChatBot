// embed.mjs — PDF embedding (knowledge.pdf, HCA.pdf)

/* ===================== Imports ===================== */
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

// CJS import of pdf-parse (do NOT add any other pdf-parse import)
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const pdfParseModule = require("pdf-parse");
const pdfParse = typeof pdfParseModule === "function"
  ? pdfParseModule
  : pdfParseModule.default;

/* ===================== Env Bootstrap ===================== */
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// Always load .env from project ROOT (../.env), even when running inside /scripts
const ENV_PATH = path.resolve(__dirname, "../.env");
dotenv.config({ path: ENV_PATH });

// Be flexible with env var names
const API_KEY =
  process.env.GOOGLE_API_KEY ||
  process.env.GEMINI_API_KEY ||
  process.env.GENAI_API_KEY;

if (!API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY (or GEMINI_API_KEY/GENAI_API_KEY) in:", ENV_PATH);
  process.exit(1);
}

console.log("📄 Using .env: ", ENV_PATH);
console.log("🔑 API key set:", "yes");

/* ===================== Paths & Config ===================== */
const DATA_DIR = path.resolve(__dirname, "..", "data");

// PDFs (env overrides). Defaults keep your old behavior.
let PDF_PATH       = process.env.PDF_PATH       || path.join(DATA_DIR, "knowledge.pdf");
let HCA_PDF_PATH   = process.env.HCA_PDF_PATH   || path.join(DATA_DIR, "HCA.pdf");

// Normalize for Windows (prefer forward slashes)
PDF_PATH       = String(PDF_PATH).replace(/\\+/g, "/");
HCA_PDF_PATH   = String(HCA_PDF_PATH).replace(/\\+/g, "/");

// Output embedding index
const OUT_PATH = process.env.OUT_PATH || path.join(DATA_DIR, "index.json");

// Embedding params
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";
const CHUNK_SIZE       = parseInt(process.env.CHUNK_SIZE      || "1200", 10);
const CHUNK_OVERLAP    = parseInt(process.env.CHUNK_OVERLAP   || "200", 10);

/* ===================== Client ===================== */
const genAI    = new GoogleGenerativeAI(API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });

/* ===================== Utilities ===================== */
function chunkText(text, size, overlap) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end === text.length) break;
    start = Math.max(0, end - overlap);
  }
  return chunks;
}

// ---------------- Stop words (EN + Hinglish + Hindi) ----------------
const EN_STOPWORDS = new Set(`
a about above after again against all am an and any are aren't as at
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
you you'd you'll you're you've your yours yourself yourselves
`.trim().split(/\s+/));

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
  "hca","hari","chand","anand","anil","duke","kansai","special","highlead","merrow","megasew","amf","reece",
  "delhi","india","solution","solutions","automation","garment","leather","mattress","dukejia","vios","jia",
  "pattern","sewing","embroidery","perforation","quilting","upholstery","fk","group","bangladesh","ethiopia"
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

/* ===================== Loaders ===================== */
async function readPdfText(filePath) {
  if (!filePath) return { ok: false, why: "no path" };
  if (!fs.existsSync(filePath)) return { ok: false, why: "missing", filePath };
  const buf = fs.readFileSync(filePath);
  if (!buf || !buf.length) return { ok: false, why: "empty", filePath };
  const parsed = await pdfParse(buf);
  const text = (parsed.text || "")
    .replace(/\r/g, "")
    .replace(/\n{2,}/g, "\n\n")
    .trim();
  return { ok: true, text, filePath };
}

/* ===================== Runner ===================== */
async function main() {
  console.log("📂 CWD:        ", process.cwd());
  console.log("📂 DATA_DIR:   ", DATA_DIR);
  console.log("📄 PDF_PATH:   ", PDF_PATH);
  console.log("📄 HCA_PDF:    ", HCA_PDF_PATH);
  console.log("📦 OUT_PATH:   ", OUT_PATH);
  console.log("🧩 CHUNK_SIZE/OVERLAP:", CHUNK_SIZE, CHUNK_OVERLAP);
  console.log("🧠 MODEL(emb): ", EMBEDDING_MODEL);

  // Collect sources (each optional; we gracefully skip if missing)
  const sources = [];

  // 1) Main PDF (knowledge.pdf or env override)
  const mainPdf = await readPdfText(PDF_PATH);
  if (mainPdf.ok) {
    sources.push({ kind: "pdf", name: path.basename(PDF_PATH), text: mainPdf.text });
    console.log("✅ Loaded:", path.basename(PDF_PATH));
  } else {
    console.log("⚠️ Skipped main PDF:", mainPdf.why, mainPdf.filePath || "");
  }

  // 2) HCA company PDF
  const hcaPdf = await readPdfText(HCA_PDF_PATH);
  if (hcaPdf.ok) {
    sources.push({ kind: "pdf", name: path.basename(HCA_PDF_PATH), text: hcaPdf.text });
    console.log("✅ Loaded:", path.basename(HCA_PDF_PATH));
  } else {
    console.log("⚠️ Skipped HCA PDF:", hcaPdf.why, hcaPdf.filePath || "");
  }

  // 3) Optional model list PDF (won't block if missing)
  const modelPdf = await readPdfText(MODEL_PDF_PATH);
  if (modelPdf.ok) {
    sources.push({ kind: "pdf", name: path.basename(MODEL_PDF_PATH), text: modelPdf.text });
    console.log("✅ Loaded:", path.basename(MODEL_PDF_PATH));
  } else {
    console.log("⚠️ Skipped model list PDF:", modelPdf.why, modelPdf.filePath || "");
  }

  if (sources.length === 0) {
    console.error("❌ No PDF sources found. Provide at least one of: PDF_PATH, HCA_PDF_PATH");
    process.exit(1);
  }

  console.log("✂️  Chunking sources…");
  const allChunks = [];
  for (const src of sources) {
    const chunks = chunkText(src.text, CHUNK_SIZE, CHUNK_OVERLAP);
    chunks.forEach((c, idx) => {
      allChunks.push({
        source: src.name,
        kind: src.kind,
        chunk_index: idx,
        text_original: c
      });
    });
    console.log(`   • ${src.name}: ${chunks.length} chunks`);
  }

  console.log(`🧠 Embedding ${allChunks.length} chunks (EN+Hinglish+Hindi stopwords)…`);
  const batchSize = 64;
  const vectors = [];

  for (let i = 0; i < allChunks.length; i += batchSize) {
    const batch = allChunks.slice(i, i + batchSize);

    const cleaned = batch.map(b => cleanForEmbedding(b.text_original) || " ");
    const res = await embedder.batchEmbedContents({
      requests: cleaned.map(text => ({
        content: { parts: [{ text }] }
      }))
    });

    const emb = res.embeddings || [];
    for (let j = 0; j < emb.length; j++) {
      const src = batch[j];
      vectors.push({
        id: i + j,
        source: src.source,
        kind: src.kind,
        chunk_index: src.chunk_index,
        text_original: src.text_original,
        text_cleaned: cleaned[j],
        embedding: emb[j].values
      });
    }
    console.log(`   → ${Math.min(i + batchSize, allChunks.length)}/${allChunks.length}`);
  }

  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  const payload = {
    createdAt: new Date().toISOString(),
    model: EMBEDDING_MODEL,
    generation_model: GENERATION_MODEL,
    stopwords: "EN+Hinglish+Hindi",
    meta: {
      sources: sources.map(s => ({ name: s.name, kind: s.kind })),
      chunk_size: CHUNK_SIZE,
      chunk_overlap: CHUNK_OVERLAP
    },
    vectors
  };
  fs.writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2));
  console.log("✅ Saved embeddings to:", OUT_PATH);
  console.log(`📊 Total vectors: ${vectors.length}`);
}

main().catch((err) => {
  console.error("⚠️ Embed error:", err?.message || err);
  process.exit(1);
});
