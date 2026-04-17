import fs from "node:fs";
import path from "node:path";
import { resolveSampleFromId } from "@/lib/sample-audio";

const INFER_API_URL = process.env.INFER_API_URL || "http://127.0.0.1:8000";
const DATASET = process.env.DATASET || "fma_large";

function trackPathFromId(trackId) {
  const six = String(trackId).padStart(6, "0");
  // NOTE: Changed this to fma_medium to match your Python script!
  return path.join(process.cwd(), "..", DATASET, six.slice(0, 3), `${six}.mp3`);
}

export async function POST(request) {
  try {
    const contentType = request.headers.get("content-type") || "";
    let formToSend = new FormData();

    // 1. Handle "Shazam Mode" Microphone Uploads
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("file");
      
      if (!file) {
        return Response.json({ error: "No audio file uploaded." }, { status: 400 });
      }
      formToSend.append("file", file, file.name);
    } 
    // 2. Handle Sample ID uploads from frontend/public/sample-audio
    //    and preserve legacy numeric trackId support for local FMA usage.
    else {
      const body = await request.json();

      if (body?.sampleId) {
        const sample = resolveSampleFromId(body.sampleId);
        if (!sample) {
          return Response.json({ error: "Sample audio file not found." }, { status: 404 });
        }

        const bytes = fs.readFileSync(sample.absolutePath);
        const blob = new Blob([bytes], { type: "audio/mpeg" });
        formToSend.append("file", blob, sample.filename);
      } else {
        const trackId = Number.parseInt(body?.trackId, 10);
        if (!Number.isFinite(trackId)) {
          return Response.json({ error: "Invalid sampleId or trackId." }, { status: 400 });
        }

        const p = trackPathFromId(trackId);
        if (!fs.existsSync(p)) {
          return Response.json({ error: "Audio file not found." }, { status: 404 });
        }

        const bytes = fs.readFileSync(p);
        const blob = new Blob([bytes], { type: "audio/mpeg" });
        formToSend.append("file", blob, path.basename(p));
      }
    }

    // Forward the FormData to the Python FastAPI backend
    // Notice we grab top_k=5 so it doesn't dilute the percentages with 0.01% micro-guesses
    const res = await fetch(`${INFER_API_URL}/predict?top_k=42`, {
      method: "POST",
      body: formToSend,
      cache: "no-store"
    });

    const data = await res.json();
    
    if (!res.ok) {
      return Response.json({ error: data?.detail || data?.error || "Model prediction failed." }, { status: res.status });
    }

    // ==========================================
    // THE FIX: COMPOSITIONAL NORMALIZATION
    // ==========================================
    if (data.predictions && Array.isArray(data.predictions)) {
      // 1. Calculate the sum of all raw probabilities returned by the model
      const totalSum = data.predictions.reduce((sum, p) => sum + p.probability, 0);
      
      // 2. Mathematically scale them so the entire pie equals exactly 1.0 (100%)
      if (totalSum > 0) {
        data.predictions = data.predictions.map(p => ({
          ...p,
          // Divide by the total sum to get the relative footprint
          probability: p.probability / totalSum
        }));
      }
    }
    // ==========================================

    return Response.json(data, { status: 200 });
  } catch (error) {
    return Response.json({ error: error.message || "Unexpected prediction error." }, { status: 500 });
  }
}
