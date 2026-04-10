import fs from "node:fs";
import { getValidationAudioAbsolutePath, getValidationAudioByTrackId } from "@/lib/validation-audio";

export async function GET(_request, { params }) {
  const { trackId: trackIdParam } = await params;
  const trackId = Number.parseInt(trackIdParam, 10);
  if (!Number.isFinite(trackId)) {
    return Response.json({ error: "Invalid track id." }, { status: 400 });
  }

  const entry = getValidationAudioByTrackId(trackId);
  if (!entry) {
    return Response.json({ error: "Track is not in bundled validation audio set." }, { status: 404 });
  }

  const p = getValidationAudioAbsolutePath(entry.filename);
  if (!fs.existsSync(p)) {
    return Response.json({ error: "Audio file not found." }, { status: 404 });
  }

  const buffer = fs.readFileSync(p);
  return new Response(buffer, {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Content-Length": String(buffer.byteLength),
      "Cache-Control": "no-store"
    }
  });
}
