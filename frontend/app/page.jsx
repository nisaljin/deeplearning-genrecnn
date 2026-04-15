"use client";

import { motion } from "framer-motion";
import { AudioUploadCard } from "@/components/ui/audio-upload-card";
import { Activity } from "lucide-react";

// Minimalist, slow-moving background frequency visualizer
function AmbientAudioGrid() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none flex items-center justify-center opacity-[0.04] z-0">
      <div 
        className="w-[120%] h-[100vh] flex items-center justify-center gap-1 sm:gap-2"
        style={{
          // Fades the lines out at the top, bottom, left, and right edges
          maskImage: "radial-gradient(ellipse at center, rgba(0,0,0,1) 10%, rgba(0,0,0,0) 70%)",
          WebkitMaskImage: "radial-gradient(ellipse at center, rgba(0,0,0,1) 10%, rgba(0,0,0,0) 70%)"
        }}
      >
        {[...Array(60)].map((_, i) => (
          <motion.div
            key={i}
            className="w-px bg-white rounded-full"
            initial={{ height: "20%" }}
            animate={{ 
              height: ["20%", `${Math.random() * 80 + 20}%`, "20%"],
              opacity: [0.3, 1, 0.3]
            }}
            transition={{
              duration: 3 + Math.random() * 4,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.1,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-black text-zinc-100 flex flex-col items-center py-16 px-4 sm:px-6 relative selection:bg-white/20 selection:text-white">
      
      {/* Immersive Audio-Wave Background */}
      <AmbientAudioGrid />

      {/* Very faint top spotlight to separate the hero from absolute darkness */}
      <div className="absolute top-0 inset-x-0 h-[50vh] bg-gradient-to-b from-white/[0.02] to-transparent pointer-events-none z-0" />

      <div className="w-full max-w-4xl flex flex-col items-center text-center z-10 mt-10 mb-16 space-y-8">
        
        {/* Minimalist Status Pill */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-white/10 bg-white/[0.02] backdrop-blur-md"
        >
          {/* <div className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-40" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-white/80" />
          </div>
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-[0.2em]">
            Music Genre Classification using Convolutional Neural Networks
          </span> */}
        </motion.div>

        {/* Hero Typography */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-4"
        >
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-medium tracking-tighter text-white leading-[1.1]">
            Listen. <span className="text-zinc-600">Analyze.</span><br />
            Understand.
          </h1>
          
          <p className="mx-auto text-lg md:text-xl text-zinc-400 max-w-2xl font-light tracking-wide leading-relaxed">
            Deploy state-of-the-art convolutional neural networks to classify audio landscapes in real-time. Run dataset fragments or broadcast live.
          </p>
        </motion.div>
      </div>

      {/* Main App Container */}
      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="z-10 w-full max-w-2xl relative group"
      >
        {/* Extremely subtle outer glow for the main module */}
        <div className="absolute -inset-1 bg-gradient-to-b from-white/10 to-transparent rounded-[2.5rem] blur-xl opacity-20 group-hover:opacity-40 transition duration-1000" />
        
        <AudioUploadCard />
      </motion.div>

      {/* Minimalist Footer */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.5 }}
        className="mt-auto pt-24 pb-8 z-10 flex items-center gap-2 text-zinc-600 text-sm font-mono"
      >
        <Activity className="w-4 h-4" />
        <span>Deep Leaning - Group 7 (Nisal, Nicolas & Kar Kin)</span>
      </motion.div>

    </main>
  );
}