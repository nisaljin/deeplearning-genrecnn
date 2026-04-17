import fs from "node:fs";
import { getAudioMimeType, resolveSampleFromId } from "@/lib/sample-audio";

export async function GET(_request, { params }) {
  const { trackId: sampleId } = await params;
  const sample = resolveSampleFromId(sampleId);
  if (!sample) {
    return Response.json({ error: "Audio file not found." }, { status: 404 });
  }

  const buffer = fs.readFileSync(sample.absolutePath);
  return new Response(buffer, {
    status: 200,
    headers: {
      "Content-Type": getAudioMimeType(sample.absolutePath),
      "Content-Length": String(buffer.byteLength),
      "Cache-Control": "no-store"
    }
  });
}
