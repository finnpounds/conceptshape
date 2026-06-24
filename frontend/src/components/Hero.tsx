"use client";

import { useUIStore } from "@/store/ui";

/**
 * Full-screen landing overlay — the first thing a visitor (or employer) sees.
 * One-line hook + two clear paths: take the guided tour, or explore freely.
 */
export default function Hero() {
  const { showHero, setShowHero, startTour, setShowAbout } = useUIStore();
  if (!showHero) return null;

  return (
    <div className="hero-overlay">
      <div className="hero-bg" aria-hidden />
      <div className="hero-content">
        <div className="hero-eyebrow">Interpretability · 3D · Transformers</div>
        <h1 className="hero-title">Semantic Geometry Explorer</h1>
        <p className="hero-hook">
          Watch meaning take physical shape inside a language model — token by
          token, layer by layer, in three dimensions.
        </p>
        <p className="hero-sub">
          A sentence enters a transformer as a handful of points. As it flows
          through the layers, each word moves, bends, and clusters. This tool
          renders that hidden motion so you can actually see it.
        </p>

        <div className="hero-actions">
          <button className="hero-btn hero-btn--primary" onClick={startTour}>
            ▶ Take the guided tour
          </button>
          <button
            className="hero-btn"
            onClick={() => setShowHero(false)}
          >
            Explore freely
          </button>
        </div>

        <button className="hero-about-link" onClick={() => setShowAbout(true)}>
          What is this? · How it works
        </button>
      </div>
    </div>
  );
}
