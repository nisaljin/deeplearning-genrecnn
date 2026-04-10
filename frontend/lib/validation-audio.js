import fs from "node:fs";
import path from "node:path";

let cachedManifest = null;
let cachedByTrackId = null;

function manifestPath() {
  return path.join(process.cwd(), "public", "validation-audio", "manifest.json");
}

export function getValidationAudioManifest() {
  if (cachedManifest) return cachedManifest;
  const json = fs.readFileSync(manifestPath(), "utf-8");
  const data = JSON.parse(json);
  cachedManifest = Array.isArray(data) ? data : [];
  cachedByTrackId = new Map(
    cachedManifest
      .filter((entry) => Number.isFinite(Number.parseInt(entry?.trackId, 10)))
      .map((entry) => [Number.parseInt(entry.trackId, 10), entry])
  );
  return cachedManifest;
}

export function getValidationAudioByTrackId(trackId) {
  getValidationAudioManifest();
  return cachedByTrackId.get(trackId) || null;
}

export function getValidationAudioAbsolutePath(filename) {
  return path.join(process.cwd(), "public", "validation-audio", filename);
}
