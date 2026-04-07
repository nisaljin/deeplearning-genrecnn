import fs from "node:fs";
import path from "node:path";

let cachedCandidates = null;

function trackPathFromId(trackId) {
  const six = String(trackId).padStart(6, "0");
  return path.join(process.cwd(), "..", "fma_large", six.slice(0, 3), `${six}.mp3`);
}

function getCandidates() {
  if (cachedCandidates) return cachedCandidates;

  const metadataPath = path.join(process.cwd(), "..", "fma_metadata", "tracks.csv");
  const csv = fs.readFileSync(metadataPath, "utf-8");
  const lines = csv.split(/\r?\n/);

  if (lines.length < 3) {
    cachedCandidates = [];
    return cachedCandidates;
  }

  const headerA = lines[0].split(",");
  const headerB = lines[1].split(",");

  const genreIdx = headerA.findIndex((h, i) => h.trim() === "track" && (headerB[i] || "").trim() === "genre_top");
  const splitIdx = headerA.findIndex((h, i) => h.trim() === "set" && (headerB[i] || "").trim() === "split");

  if (genreIdx < 0 || splitIdx < 0) {
    cachedCandidates = [];
    return cachedCandidates;
  }

  const out = [];
  for (let i = 2; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line) continue;
    const cols = line.split(",");
    const rawId = Number.parseInt(cols[0], 10);
    if (!Number.isFinite(rawId)) continue;

    const split = (cols[splitIdx] || "").replaceAll('"', "").trim();
    const genre = (cols[genreIdx] || "").replaceAll('"', "").trim();
    if (!genre) continue;
    if (split !== "validation" && split !== "test") continue;

    const filePath = trackPathFromId(rawId);
    if (!fs.existsSync(filePath)) continue;

    out.push({ trackId: rawId, split, genre, filename: path.basename(filePath) });
  }

  cachedCandidates = out;
  return cachedCandidates;
}

export async function GET() {
  try {
    const candidates = getCandidates();
    if (!candidates.length) {
      return Response.json({ error: "No validation/test audio candidates found." }, { status: 404 });
    }
    const sample = candidates[Math.floor(Math.random() * candidates.length)];
    return Response.json(sample, { status: 200 });
  } catch (error) {
    return Response.json({ error: error.message || "Failed to choose random audio." }, { status: 500 });
  }
}
