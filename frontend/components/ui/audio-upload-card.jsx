"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Shuffle, Sparkles, Mic, Square, FileAudio, Disc3, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

// Refined, high-end muted colors for a professional look
const GENRE_COLORS = {
  Electronic: "bg-cyan-400/80 shadow-[0_0_10px_rgba(34,211,238,0.3)]",
  Rock: "bg-rose-500/80 shadow-[0_0_10px_rgba(244,63,94,0.3)]",
  "Hip-Hop": "bg-amber-500/80 shadow-[0_0_10px_rgba(245,158,11,0.3)]",
  Folk: "bg-emerald-400/80 shadow-[0_0_10px_rgba(52,211,153,0.3)]",
  Pop: "bg-fuchsia-400/80 shadow-[0_0_10px_rgba(232,121,249,0.3)]",
  Instrumental: "bg-indigo-400/80 shadow-[0_0_10px_rgba(129,140,248,0.3)]",
  Experimental: "bg-violet-500/80 shadow-[0_0_10px_rgba(139,92,246,0.3)]",
  International: "bg-yellow-400/80 shadow-[0_0_10px_rgba(250,204,21,0.3)]",
  Classical: "bg-teal-400/80 shadow-[0_0_10px_rgba(45,212,191,0.3)]",
  Jazz: "bg-orange-400/80 shadow-[0_0_10px_rgba(251,146,60,0.3)]",
  default: "bg-zinc-400 shadow-[0_0_10px_rgba(161,161,170,0.3)]"
};

// Sleek, Circular Radial Audio Visualizer
function LiveAudioVisualizer({ stream }) {
  const barsRef = useRef([]);
  const animationRef = useRef(null);
  const NUM_BARS = 40; // Gives a very smooth, dense circle

  useEffect(() => {
    if (!stream) return;

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    
    analyser.fftSize = 128; 
    source.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const updateBars = () => {
      analyser.getByteFrequencyData(dataArray);
      
      barsRef.current.forEach((bar, index) => {
        if (bar && dataArray[index] !== undefined) {
          const value = dataArray[index];
          // Map frequency (0-255) to a height (4px to 32px) extending outwards
          const height = 4 + (value / 255) * 32;
          bar.style.height = `${height}px`;
          // Add a subtle brightness effect based on volume
          bar.style.opacity = 0.3 + (value / 255) * 0.7;
        }
      });

      animationRef.current = requestAnimationFrame(updateBars);
    };

    updateBars();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (audioContext.state !== 'closed') audioContext.close();
    };
  }, [stream]);

  return (
    <div className="relative w-40 h-40 flex items-center justify-center my-2">
      {/* Central Inner Pulse */}
      <div className="absolute inset-0 m-auto w-16 h-16 rounded-full border border-white/10 bg-white/[0.02] shadow-[0_0_30px_rgba(255,255,255,0.05)] animate-pulse flex items-center justify-center">
        <Activity className="w-6 h-6 text-zinc-500 animate-pulse" />
      </div>

      {/* Radial Waveform Bars */}
      {[...Array(NUM_BARS)].map((_, i) => (
        <div
          key={i}
          ref={el => barsRef.current[i] = el}
          className="absolute bg-white rounded-full transition-[height] duration-75 ease-out"
          style={{
            top: '50%',
            left: '50%',
            width: '3px',
            height: '4px',
            transformOrigin: '50% 0%',
            // Rotate each bar, then push it outwards from the center by 38px
            transform: `translate(-50%, 0%) rotate(${i * (360 / NUM_BARS)}deg) translateY(38px)`
          }}
        />
      ))}
    </div>
  );
}

function PredictionList({ predictions }) {
  if (!predictions?.length) return null;

  return (
    <motion.div 
      initial={{ opacity: 0, height: 0, marginTop: 0 }} 
      animate={{ opacity: 1, height: "auto", marginTop: 24 }} 
      exit={{ opacity: 0, height: 0 }}
      className="space-y-4 text-left p-6 bg-black/40 rounded-2xl border border-white/5 backdrop-blur-xl"
    >
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-zinc-400" />
        <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest">Analysis Results</h3>
      </div>
      {predictions.map((p, idx) => {
        const percent = (p.probability * 100).toFixed(1);
        const colorClass = GENRE_COLORS[p.genre] || GENRE_COLORS.default;
        
        return (
          <div key={`${p.genre}-${idx}`} className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium text-zinc-200 capitalize tracking-wide">{p.genre}</span>
              <span className="text-zinc-500 font-mono">{percent}%</span>
            </div>
            <div className="h-1.5 w-full bg-zinc-900 rounded-full overflow-hidden backdrop-blur-sm">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${percent}%` }}
                transition={{ duration: 1.2, delay: idx * 0.15, ease: [0.16, 1, 0.3, 1] }} 
                className={cn("h-full rounded-full", colorClass)}
              />
            </div>
          </div>
        );
      })}
    </motion.div>
  );
}

export function AudioUploadCard({ className }) {
  const [mode, setMode] = useState("dataset");
  
  const [sample, setSample] = useState(null);
  const [loadingSample, setLoadingSample] = useState(true);
  
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [activeStream, setActiveStream] = useState(null); 
  
  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);
  const stopTimeoutRef = useRef(null);
  const chunksRef = useRef([]);

  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  const fetchRandomSample = async () => {
    try {
      setLoadingSample(true); setError(""); setPredictions([]);
      const res = await fetch("/api/random-audio", { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to fetch random sample");
      setSample(await res.json());
    } catch (e) { setError(e.message || "Error sampling audio"); } 
    finally { setLoadingSample(false); }
  };

  useEffect(() => { fetchRandomSample(); }, []);

  const startRecording = async () => {
    try {
      // 1. Get the raw, unprocessed hardware stream
      const rawStream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          echoCancellation: false, 
          noiseSuppression: false, 
          autoGainControl: false, 
          channelCount: 2 
        } 
      });

      // 2. Set up an Audio Processing Pipeline
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(rawStream);

      // 3. Add a High-Pass Filter (Cuts out sub-bass rumble, wind, and table noise below 70Hz)
      const filterNode = audioContext.createBiquadFilter();
      filterNode.type = "highpass";
      filterNode.frequency.value = 70; 

      // 4. Add a Gain Node to manually boost the volume (Multiplier)
      const gainNode = audioContext.createGain();
      gainNode.gain.value = 2.5; // <-- 2.5x volume boost. You can adjust this between 1.5 and 4.0 if needed.

      // 5. Create a destination for the processed stream
      const destination = audioContext.createMediaStreamDestination();

      // Connect the pipeline: Source -> Filter -> Gain -> Destination
      source.connect(filterNode);
      filterNode.connect(gainNode);
      gainNode.connect(destination);

      const processedStream = destination.stream;

      // Feed the PROCESSED stream to the visualizer so the UI reflects the louder volume
      setActiveStream(processedStream);

      // Feed the PROCESSED stream into the MediaRecorder
      const mediaRecorder = new MediaRecorder(processedStream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecordedAudio({ blob, url: URL.createObjectURL(blob), mimeType });
        
        // Clean up both streams and the audio context to prevent memory leaks
        rawStream.getTracks().forEach(t => t.stop());
        processedStream.getTracks().forEach(t => t.stop());
        if (audioContext.state !== "closed") audioContext.close();
        
        setActiveStream(null); 
        clearInterval(timerRef.current); 
        clearTimeout(stopTimeoutRef.current); 
        setIsRecording(false);
      };

      mediaRecorder.start();
      setIsRecording(true); setRecordingTime(0); setError(""); setPredictions([]); setRecordedAudio(null);
      timerRef.current = setInterval(() => setRecordingTime((p) => Math.min(p + 1, 30)), 1000);
      stopTimeoutRef.current = setTimeout(() => stopRecording(), 30000);
    } catch (err) { setError("Microphone access denied or audio hardware unavailable."); }
  };

  const stopRecording = () => { if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop(); };

  const runPrediction = async () => {
    try {
      setPredicting(true); setError(""); setPredictions([]);
      let res;
      if (mode === "dataset") {
        if (!sample?.trackId) return;
        res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trackId: sample.trackId }) });
      } else {
        if (!recordedAudio?.blob) return;
        const formData = new FormData();
        formData.append("file", recordedAudio.blob, `recording.${recordedAudio.mimeType.includes('mp4') ? 'm4a' : 'webm'}`);
        res = await fetch("/api/predict", { method: "POST", body: formData });
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setPredictions(data.predictions || []);
    } catch (e) { setError(e.message || "Error during prediction"); } 
    finally { setPredicting(false); }
  };

  return (
    <motion.div
      layout
      className={cn("relative w-full rounded-[2rem] bg-zinc-950/60 backdrop-blur-3xl border border-white/10 shadow-2xl overflow-hidden", className)}
    >
      {/* Very faint pure white gradient glow, stripping out the pinks/purples */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.03] via-transparent to-white/[0.01] pointer-events-none" />

      {/* Sleek Tab Bar */}
      <div className="p-2 w-full">
        <div className="flex relative bg-black/40 rounded-full p-1 backdrop-blur-md border border-white/5">
          {["dataset", "record"].map((tab) => (
            <button
              key={tab}
              onClick={() => { setMode(tab); setPredictions([]); setError(""); }}
              className="relative flex-1 py-2.5 text-sm font-medium rounded-full outline-none z-10"
            >
              {mode === tab && (
                <motion.div
                  layoutId="active-tab"
                  className="absolute inset-0 bg-zinc-800/90 rounded-full shadow-lg border border-white/10"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
              <span className={cn("relative z-20 transition-colors duration-300", mode === tab ? "text-white" : "text-zinc-500 hover:text-zinc-300")}>
                {tab === "dataset" ? "Dataset Library" : "Shazam Mode"}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 sm:p-8 relative z-10">
        <AnimatePresence mode="wait">
          {/* DATASET MODE */}
          {mode === "dataset" && (
            <motion.div key="dataset" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }} className="space-y-6">
              <div className="group relative flex items-center justify-between p-5 bg-zinc-900/40 rounded-2xl border border-white/5 overflow-hidden transition-all hover:border-white/10">
                <div className="flex items-center gap-4 z-10">
                  <motion.div 
                    animate={{ rotate: loadingSample ? 360 : 0 }} 
                    transition={{ repeat: loadingSample ? Infinity : 0, duration: 2, ease: "linear" }}
                    className="flex-shrink-0 w-12 h-12 rounded-full bg-zinc-800 border border-white/10 flex items-center justify-center shadow-lg"
                  >
                    <Disc3 className={cn("w-5 h-5", loadingSample ? "text-zinc-500" : "text-zinc-300")} />
                  </motion.div>
                  <div>
                    <p className="text-[10px] text-zinc-500 font-mono tracking-widest uppercase mb-1">Track #{sample?.trackId || "---"}</p>
                    <p className="font-medium text-zinc-200 text-base leading-none">{loadingSample ? "Querying database..." : sample?.filename || "Unknown Track"}</p>
                  </div>
                </div>
                
                <motion.button 
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={fetchRandomSample} disabled={loadingSample || predicting} 
                  className="z-10 p-3 rounded-full bg-white/5 hover:bg-white/10 text-zinc-300 transition-all disabled:opacity-50 border border-white/5"
                >
                  {loadingSample ? <Loader2 className="h-4 w-4 animate-spin" /> : <Shuffle className="h-4 w-4" />}
                </motion.button>
              </div>

              {sample && (
                <div className="relative rounded-2xl overflow-hidden border border-white/5 bg-black/40 p-2 backdrop-blur-xl">
                  <audio className="w-full h-10 outline-none [&::-webkit-media-controls-panel]:bg-transparent [&::-webkit-media-controls-current-time-display]:text-zinc-300 [&::-webkit-media-controls-time-remaining-display]:text-zinc-300" controls src={`/api/audio/${sample.trackId}`} />
                </div>
              )}
            </motion.div>
          )}

          {/* RECORD MODE */}
          {mode === "record" && (
            <motion.div key="record" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} className="flex flex-col items-center justify-center min-h-[220px]">
              
              {!isRecording && !recordedAudio && (
                <motion.button 
                  whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} 
                  onClick={startRecording} 
                  className="relative flex flex-col items-center gap-6 group"
                >
                  <div className="relative h-24 w-24 rounded-full bg-zinc-900 border border-white/10 flex items-center justify-center shadow-xl group-hover:bg-white/5 group-hover:border-white/20 transition-all z-10">
                    <Mic className="h-8 w-8 text-zinc-400 group-hover:text-white transition-colors" />
                  </div>
                  <p className="text-xs font-medium text-zinc-500 group-hover:text-zinc-300 tracking-widest uppercase">Tap to Record</p>
                </motion.button>
              )}

              {isRecording && (
                <div className="w-full flex flex-col items-center space-y-6">
                  {/* High-End Circular Visualizer Mounted Here */}
                  <LiveAudioVisualizer stream={activeStream} />
                  
                  <div className="text-center">
                    <p className="text-2xl font-mono text-zinc-300 font-light tracking-widest">
                      0:{String(recordingTime).padStart(2, '0')}
                    </p>
                  </div>
                  <motion.button 
                    whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                    onClick={stopRecording} 
                    className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-zinc-800 hover:bg-zinc-700 border border-white/10 text-white text-sm font-medium transition-all"
                  >
                    <Square className="h-3.5 w-3.5 fill-current text-red-500" /> Stop Recording
                  </motion.button>
                </div>
              )}

              {!isRecording && recordedAudio && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="w-full space-y-6">
                  <div className="p-4 bg-zinc-900/40 border border-white/5 rounded-2xl flex items-center gap-4 backdrop-blur-xl">
                    <div className="p-2.5 bg-white/5 rounded-full text-zinc-300"><FileAudio className="h-5 w-5" /></div>
                    <div className="flex-1">
                      <p className="text-zinc-200 font-medium text-sm">Captured Audio</p>
                      <p className="text-xs text-zinc-500 font-mono mt-0.5">{recordingTime}s duration</p>
                    </div>
                    <button onClick={() => { setRecordedAudio(null); setPredictions([]); }} className="text-xs font-medium text-zinc-500 hover:text-white transition-colors">
                      Discard
                    </button>
                  </div>
                  <div className="relative rounded-2xl overflow-hidden border border-white/5 bg-black/40 p-2 backdrop-blur-xl">
                    <audio className="w-full h-10 outline-none [&::-webkit-media-controls-panel]:bg-transparent" controls src={recordedAudio.url} />
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Minimalist High-Contrast Analyze Button */}
        {((mode === "dataset" && sample) || (mode === "record" && recordedAudio && !isRecording)) && (
          <motion.button 
            layout
            whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
            onClick={runPrediction} disabled={predicting} 
            className="mt-8 w-full flex items-center justify-center gap-3 py-3.5 rounded-xl bg-white hover:bg-zinc-200 text-black font-semibold transition-all shadow-lg disabled:opacity-50"
          >
            {predicting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            <span className="tracking-wide text-sm">{predicting ? "Analyzing Topology..." : "Detect Genre"}</span>
          </motion.button>
        )}

        <AnimatePresence>
          {error && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-6 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-center">
              <p className="text-xs font-medium text-red-400">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        <PredictionList predictions={predictions} />
      </div>
    </motion.div>
  );
}