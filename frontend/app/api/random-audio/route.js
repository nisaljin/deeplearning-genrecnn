import { listSampleAudioFiles } from "@/lib/sample-audio";

export async function GET() {
  try {
    const candidates = listSampleAudioFiles();
    if (!candidates.length) {
      return Response.json(
        { error: "No sample audio files found in frontend/public/sample-audio." },
        { status: 404 }
      );
    }
    const sample = candidates[Math.floor(Math.random() * candidates.length)];
    return Response.json(sample, { status: 200 });
  } catch (error) {
    return Response.json({ error: error.message || "Failed to choose random audio." }, { status: 500 });
  }
}
