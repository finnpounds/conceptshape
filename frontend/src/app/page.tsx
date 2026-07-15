"use client";

import dynamic from "next/dynamic";
import Controls from "@/components/Controls";
import CorpusControls from "@/components/CorpusControls";
import SongControls from "@/components/SongControls";
import TrainingControls from "@/components/TrainingControls";
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
const TrainingViewport = dynamic(() => import("@/components/TrainingViewport"), { ssr: false });

export default function Home() {
  const { appMode, setAppMode, startTour, setShowAbout, sidebarOpen, setSidebarOpen } =
    useUIStore();

  return (
    <main className="app-layout">
      <div className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="logo-area">
          <button
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close controls"
          >
            ✕
          </button>
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
            <button
              className={`method-btn ${appMode === "training" ? "active" : ""}`}
              onClick={() => setAppMode("training")}
              title="Watch concept geometry crystallize across 300B tokens of training"
            >
              Training
            </button>
          </div>

          <BackendStatus />
        </div>

        {appMode === "explore" ? <Controls /> :
         appMode === "corpus" ? <CorpusControls /> :
         appMode === "song" ? <SongControls /> :
         <TrainingControls />}
      </div>

      <div className="viewport-area">
        {appMode === "explore" ? <Viewport /> :
         appMode === "corpus" ? <CorpusViewport /> :
         appMode === "song" ? <SongViewport /> :
         <TrainingViewport />}

        {/* Showcase overlays live above the active viewport */}
        <Tour />
      </div>

      {/* Mobile: floating button to open the controls drawer + backdrop */}
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open controls"
      >
        ☰ Controls
      </button>
      <div
        className={`mobile-backdrop ${sidebarOpen ? "show" : ""}`}
        onClick={() => setSidebarOpen(false)}
        aria-hidden
      />

      {/* Full-screen overlays */}
      <Hero />
      <AboutPanel />
    </main>
  );
}
