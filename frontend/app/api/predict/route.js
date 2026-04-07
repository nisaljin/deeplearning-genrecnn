import fs from "node:fs";
import path from "node:path";

const INFER_API_URL = process.env.INFER_API_URL || "http://127.0.0.1:8000";

function trackPathFromId(trackId) {
  const six = String(trackId).padStart(6, "0");
  return path.join(process.cwd(), "..", "fma_large", six.slice(0, 3), `${six}.mp3`);
}

export async function POST(request) {
  try {
    const body = await request.json();
    const trackId = Number.parseInt(body?.trackId, 10);
    if (!Number.isFinite(trackId)) {
      return Response.json({ error: "Invalid track id." }, { status: 400 });
    }

    const p = trackPathFromId(trackId);
    if (!fs.existsSync(p)) {
      return Response.json({ error: "Audio file not found." }, { status: 404 });
    }

    const bytes = fs.readFileSync(p);
    const blob = new Blob([bytes], { type: "audio/mpeg" });
    const form = new FormData();
    form.append("file", blob, path.basename(p));

    const res = await fetch(`${INFER_API_URL}/predict?top_k=5`, {
      method: "POST",
      body: form,
      cache: "no-store"
    });

    const data = await res.json();
    if (!res.ok) {
      return Response.json({ error: data?.detail || data?.error || "Model prediction failed." }, { status: res.status });
    }

    return Response.json(data, { status: 200 });
  } catch (error) {
    return Response.json({ error: error.message || "Unexpected prediction error." }, { status: 500 });
  }
}
