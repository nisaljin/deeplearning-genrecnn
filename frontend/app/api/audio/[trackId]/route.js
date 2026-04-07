import fs from "node:fs";
import path from "node:path";

function trackPathFromId(trackId) {
  const six = String(trackId).padStart(6, "0");
  return path.join(process.cwd(), "..", "fma_large", six.slice(0, 3), `${six}.mp3`);
}

export async function GET(_request, { params }) {
  const { trackId: trackIdParam } = await params;
  const trackId = Number.parseInt(trackIdParam, 10);
  if (!Number.isFinite(trackId)) {
    return Response.json({ error: "Invalid track id." }, { status: 400 });
  }

  const p = trackPathFromId(trackId);
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
