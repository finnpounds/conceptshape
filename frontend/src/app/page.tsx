"use client";

import dynamic from "next/dynamic";
import Controls from "@/components/Controls";
import CorpusControls from "@/components/CorpusControls";
import SongControls from "@/components/SongControls";
import Hero from "@/components/Hero";
import Tour from "@/components/Tour";
import AboutPanel from "@/components/AboutPanel";
import BackendStatus from "@/components/BackendStatus";
import { useUIStore } from "@/store/ui";
import { GITHUB_URL } from "@/lib/config";

// R3F must be loaded client-side only
const Viewport = dynamic(() => import("@/components/Viewport"), { ssr: false });
const CorpusViewport = dynamic(() => import("@/components/CorpusViewport"), { ssr: false });
const SongViewport = dynamic(() => import("@/components/SongViewport"), { ssr: false });

export default function Home() {
  const { appMode, setAppMode, startTour, setShowAbout } = useUIStore();

  return (
    <main className="app-layout">
      <div className="sidebar">
        <div className="logo-area">
          <h1 className="app-title">Semantic Geometry</h1>
          <p className="app-subtitle">
            Watch meaning take shape through transformer layers
          </p>

          {/* Showcase nav */}
          <div className="nav-row">
            <button className="nav-btn" onClick={startTour} title="Guided walkthrough">
              ▶ Tour
            </button>
            <button className="nav-btn" onClick={() => setShowAbout(true)} title="What is this?">
              About
            </button>
            <a className="nav-btn" href={GITHUB_URL} target="_blank" rel="noopener noreferrer" title="Source code">
              GitHub ↗
            </a>
          </div>

          {/* Top-level mode toggle */}
          <div className="method-toggle" style={{ marginTop: 10 }}>
            <button
              className={`method-btn ${appMode === "explore" ? "active" : ""}`}
              onClick={() => setAppMode("explore")}
              title="Single-sentence trajectory explorer"
            >
              Explore
            </button>
            <button
              className={`method-btn ${appMode === "corpus" ? "active" : ""}`}
              onClick={() => setAppMode("corpus")}
              title="Corpus-scale topological shape comparison"
            >
              Corpus
            </button>
            <button
              className={`method-btn ${appMode === "song" ? "active" : ""}`}
              onClick={() => setAppMode("song")}
              title="Song lyrics visualizer — meaning animating through layers"
            >
              Song
            </button>
          </div>

          <BackendStatus />
        </div>

        {appMode === "explore" ? <Controls /> :
         appMode === "corpus" ? <CorpusControls /> :
         <SongControls />}
      </div>

      <div className="viewport-area">
        {appMode === "explore" ? <Viewport /> :
         appMode === "corpus" ? <CorpusViewport /> :
         <SongViewport />}

        {/* Showcase overlays live above the active viewport */}
        <Tour />
      </div>

      {/* Full-screen overlays */}
      <Hero />
      <AboutPanel />
    </main>
  );
}
