"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Shuffle, WandSparkles } from "lucide-react";
import { cn } from "@/lib/utils";

function Waveform({ width = 220, height = 36, bars = 36 }) {
  const [barsArray, setBarsArray] = useState([]);

  useEffect(() => {
    const barWidth = (width / bars) * 0.5;
    const spacing = (width / bars) * 0.4;
    const centerY = height / 2;
    const radius = barWidth * 0.7;
    const totalBarsWidth = (barWidth + spacing) * bars - spacing;
    const startX = (width - totalBarsWidth) / 2;

    const arr = [];
    for (let i = 0; i < bars; i += 1) {
      const x = startX + i * (barWidth + spacing);
      const barHeight = Math.random() * (height * 0.6) + height * 0.1;
      const topY = centerY - barHeight / 2;
      arr.push(
        <rect
          key={i}
          x={x}
          y={topY}
          width={barWidth}
          height={barHeight}
          rx={radius}
          ry={radius}
          fill="currentColor"
          className="text-muted-foreground/60"
        />
      );
    }
    setBarsArray(arr);
  }, [width, height, bars]);

  return <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>{barsArray}</svg>;
}

function PredictionList({ predictions }) {
  if (!predictions?.length) return null;

  return (
    <div className="mt-4 space-y-2 text-left">
      {predictions.map((p, idx) => (
        <div key={`${p.genre}-${idx}`} className="rounded-md bg-muted/70 px-3 py-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{p.genre}</span>
            <span>{(p.probability * 100).toFixed(2)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function AudioUploadCard({
  className,
  title = "Unseen Audio Demo",
  description = "Randomly samples bundled validation audio and predicts genre."
}) {
  const [sample, setSample] = useState(null);
  const [loadingSample, setLoadingSample] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  const fetchRandomSample = async () => {
    try {
      setLoadingSample(true);
      setError("");
      setPredictions([]);
      const res = await fetch("/api/random-audio", { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to fetch random sample");
      const data = await res.json();
      setSample(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error while sampling audio");
    } finally {
      setLoadingSample(false);
    }
  };

  useEffect(() => {
    fetchRandomSample();
  }, []);

  const runPrediction = async () => {
    if (!sample?.trackId) return;
    try {
      setPredicting(true);
      setError("");
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trackId: sample.trackId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setPredictions(data.predictions || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error during prediction");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <motion.div
      className={cn("relative w-full max-w-xl mx-auto", className)}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 220, damping: 26 }}
    >
      <div className="rounded-xl border border-border/60 bg-card/95 p-6 shadow-lg">
        <div className="space-y-6">
          <div className="text-left">
            <h2 className="text-xl font-semibold">{title}</h2>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>

          <div className="rounded-xl border-2 border-dashed border-border/70 bg-muted/40 p-5">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium">{loadingSample ? "Picking unseen sample..." : sample?.filename || "No sample"}</p>
                <p className="text-xs text-muted-foreground">
                  {sample ? `Track #${sample.trackId} • split: ${sample.split} • genre: ${sample.genre}` : "Bundled validation set"}
                </p>
              </div>

              <button
                onClick={fetchRandomSample}
                disabled={loadingSample || predicting}
                className="inline-flex items-center gap-2 rounded-md bg-secondary px-3 py-2 text-sm hover:bg-secondary/80 disabled:opacity-50"
              >
                {loadingSample ? <Loader2 className="h-4 w-4 animate-spin" /> : <Shuffle className="h-4 w-4" />} Random
              </button>
            </div>

            <div className="mt-4 flex items-center justify-center rounded-lg bg-card p-3">
              <Waveform />
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={runPrediction}
                disabled={!sample || predicting || loadingSample}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                {predicting ? <Loader2 className="h-4 w-4 animate-spin" /> : <WandSparkles className="h-4 w-4" />} Predict
              </button>
            </div>

            {sample ? (
              <audio className="mt-4 w-full" controls src={`/api/audio/${sample.trackId}`}>
                Your browser does not support audio playback.
              </audio>
            ) : null}
          </div>

          <AnimatePresence>
            {error ? (
              <motion.p
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-sm text-destructive"
              >
                {error}
              </motion.p>
            ) : null}
          </AnimatePresence>

          <PredictionList predictions={predictions} />
        </div>
      </div>
    </motion.div>
  );
}
