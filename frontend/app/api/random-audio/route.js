import { getValidationAudioManifest } from "@/lib/validation-audio";

let cachedCandidates = null;

function getCandidates() {
  if (cachedCandidates) return cachedCandidates;
  const manifest = getValidationAudioManifest();
  cachedCandidates = manifest;
  return cachedCandidates;
}

export async function GET() {
  try {
    const candidates = getCandidates();
    if (!candidates.length) {
      return Response.json({ error: "No bundled validation audio candidates found." }, { status: 404 });
    }
    const sample = candidates[Math.floor(Math.random() * candidates.length)];
    return Response.json(sample, { status: 200 });
  } catch (error) {
    return Response.json({ error: error.message || "Failed to choose random audio." }, { status: 500 });
  }
}
