import "dotenv/config";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.resolve(__dirname, "..", "data");

// Resolve PDF path: .env > default data/knowledge.pdf
let PDF_PATH = process.env.PDF_PATH || path.join(DATA_DIR, "knowledge.pdf");
// Normalize for Windows (prefer forward slashes)
PDF_PATH = PDF_PATH.replace(/\\+/g, "/");

const OUT_PATH = path.join(DATA_DIR, "index.json");
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE || "1200", 10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || "200", 10);

if (!process.env.GOOGLE_API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });

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
  "hca","hari","chand","anand","anil","anand","duke","kansai","special","highlead","merrow","megasew","amf","reece",
  "delhi","india","solution","solutions","automation","garment","leather","mattress"
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

async function main() {
  console.log("📂 CWD:", process.cwd());
  console.log("📂 DATA_DIR:", DATA_DIR);
  console.log("📄 PDF_PATH:", PDF_PATH);
  console.log("💾 OUT_PATH:", OUT_PATH);

  if (!fs.existsSync(PDF_PATH)) {
    console.error("❌ PDF not found at:", PDF_PATH);
    console.error("👉 Place file at data/knowledge.pdf OR set PDF_PATH in .env (use forward slashes).");
    process.exit(1);
  }

  // Read as BUFFER — so pdf-parse never tries its test PDF
  const pdfBuffer = fs.readFileSync(PDF_PATH);
  if (!pdfBuffer || !pdfBuffer.length) {
    console.error("❌ Could not read PDF (buffer empty). Check file permissions/path.");
    process.exit(1);
  }

  const parsed = await pdfParse(pdfBuffer);
  const text = parsed.text.replace(/\r/g, "").replace(/\n{2,}/g, "\n\n").trim();

  console.log("✂️  Chunking…");
  const chunks = chunkText(text, CHUNK_SIZE, CHUNK_OVERLAP);

  console.log(`🧠 Embedding ${chunks.length} chunks (EN+Hinglish+Hindi stopwords)…`);
  const batchSize = 64;
  const vectors = [];

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batchOriginal = chunks.slice(i, i + batchSize);
    const batchCleaned = batchOriginal.map(c => cleanForEmbedding(c));
    const res = await embedder.batchEmbedContents({
      requests: batchCleaned.map((c) => ({
        content: { parts: [{ text: c || " " }] }
      }))
    });

    // NOTE: result field is `embeddings` (array) with `.values`
    const batchVectors = res.embeddings.map((e, j) => ({
      id: i + j,
      text_original: batchOriginal[j],
      text_cleaned: batchCleaned[j],
      embedding: e.values
    }));

    vectors.push(...batchVectors);
    console.log(`   → ${Math.min(i + batchSize, chunks.length)}/${chunks.length}`);
  }

  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify({
      createdAt: new Date().toISOString(),
      model: EMBEDDING_MODEL,
      stopwords: "EN+Hinglish+Hindi",
      vectors
    }, null, 2)
  );
  console.log("✅ Saved embeddings to:", OUT_PATH);
}

main().catch((err) => {
  console.error("⚠️ Embed error:", err?.message || err);
  process.exit(1);
});
