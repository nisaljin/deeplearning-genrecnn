import fs from "node:fs";
import path from "node:path";

const AUDIO_EXTENSIONS = new Set([".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm", ".weba", ".mp4"]);

const MIME_BY_EXTENSION = {
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".m4a": "audio/mp4",
  ".mp4": "audio/mp4",
  ".webm": "audio/webm",
  ".weba": "audio/webm"
};

const EXCLUDED_SAMPLE_RELATIVE_PATHS = new Set([
  path.join("prominent", "experimental", "000137.mp3"),
  path.join("prominent", "experimental", "000138.mp3")
]);

export function getSampleAudioDir() {
  return path.join(process.cwd(), "public", "sample-audio");
}

function walkAudioFiles(rootDir, subDir = "") {
  const currentDir = path.join(rootDir, subDir);
  const entries = fs.readdirSync(currentDir, { withFileTypes: true });
  const out = [];

  for (const entry of entries) {
    const relPath = subDir ? path.join(subDir, entry.name) : entry.name;
    if (entry.isDirectory()) {
      out.push(...walkAudioFiles(rootDir, relPath));
      continue;
    }

    const ext = path.extname(entry.name).toLowerCase();
    if (!AUDIO_EXTENSIONS.has(ext)) continue;
    if (EXCLUDED_SAMPLE_RELATIVE_PATHS.has(relPath)) continue;

    const sampleId = Buffer.from(relPath).toString("base64url");
    const encodedUrlPath = relPath
      .split(path.sep)
      .map((segment) => encodeURIComponent(segment))
      .join("/");

    out.push({
      sampleId,
      filename: entry.name,
      relativePath: relPath,
      fileUrl: `/sample-audio/${encodedUrlPath}`
    });
  }

  return out;
}

export function listSampleAudioFiles() {
  const sampleDir = getSampleAudioDir();
  if (!fs.existsSync(sampleDir)) return [];
  return walkAudioFiles(sampleDir).sort((a, b) => a.relativePath.localeCompare(b.relativePath));
}

function splitSegments(relativePath) {
  return relativePath
    .split(path.sep)
    .map((segment) => segment.trim())
    .filter(Boolean);
}

const CURATED_TIERS = new Set(["prominent", "minority"]);

export function listCuratedGenreAudioFiles() {
  const allSamples = listSampleAudioFiles();
  return allSamples
    .map((sample) => {
      const segments = splitSegments(sample.relativePath);
      if (segments.length < 3) return null;
      const tier = segments[0];
      if (!CURATED_TIERS.has(tier)) return null;
      const genre = segments[1];
      if (!genre) return null;
      return {
        ...sample,
        tier,
        genre
      };
    })
    .filter(Boolean);
}

export function listBestGenreAudioFiles() {
  // Backward compatible alias used by existing API routes.
  return listCuratedGenreAudioFiles();
}

export function resolveSampleFromId(sampleId) {
  if (!sampleId || typeof sampleId !== "string") return null;

  let relativePath;
  try {
    relativePath = Buffer.from(sampleId, "base64url").toString("utf-8");
  } catch {
    return null;
  }

  if (!relativePath || relativePath.includes("\0")) return null;
  if (EXCLUDED_SAMPLE_RELATIVE_PATHS.has(relativePath)) return null;

  const sampleDir = getSampleAudioDir();
  const absolutePath = path.resolve(sampleDir, relativePath);
  const sampleDirResolved = path.resolve(sampleDir);

  if (!absolutePath.startsWith(`${sampleDirResolved}${path.sep}`)) return null;
  if (!fs.existsSync(absolutePath)) return null;

  return {
    relativePath,
    absolutePath,
    filename: path.basename(absolutePath)
  };
}

export function getAudioMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_BY_EXTENSION[ext] || "application/octet-stream";
}
