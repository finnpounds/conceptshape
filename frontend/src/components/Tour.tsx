"use client";

import { useEffect, useState } from "react";
import { useUIStore } from "@/store/ui";
import {
  fetchGalleryIndex,
  applyExample,
  type GalleryEntry,
} from "@/lib/gallery";

/**
 * Guided tour. A floating caption card walks through the curated gallery
 * examples; each step loads precomputed data into the live 3D scene behind it,
 * so the visitor sees the real visualization while reading a plain-English
 * explanation. Fully interactive — they can orbit the scene at any step.
 */
export default function Tour() {
  const { tourIndex, setTourIndex, endTour } = useUIStore();
  const [examples, setExamples] = useState<GalleryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the manifest once the tour starts.
  useEffect(() => {
    if (tourIndex === null || examples.length > 0) return;
    fetchGalleryIndex()
      .then((idx) => setExamples(idx.examples))
      .catch((e) => setError(e.message));
  }, [tourIndex, examples.length]);

  // Apply the current step's data whenever the step (or loaded set) changes.
  useEffect(() => {
    if (tourIndex === null || examples.length === 0) return;
    const entry = examples[tourIndex];
    if (!entry) return;
    setLoading(true);
    setError(null);
    applyExample(entry)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [tourIndex, examples]);

  if (tourIndex === null) return null;

  const entry = examples[tourIndex];
  const total = examples.length;

  return (
    <div className="tour-card">
      <div className="tour-card-head">
        <span className="tour-step">
          {total > 0 ? `${tourIndex + 1} / ${total}` : "…"}
        </span>
        <span className="tour-mode-tag">
          {entry ? `${entry.mode} · ${entry.subMode}` : ""}
        </span>
        <button className="tour-exit" onClick={endTour} title="Exit tour">
          ✕
        </button>
      </div>

      {error ? (
        <p className="tour-error">Couldn’t load this example: {error}</p>
      ) : entry ? (
        <>
          <h2 className="tour-title">
            {entry.title}
            {loading && <span className="tour-loading"> · loading…</span>}
          </h2>
          <p className="tour-hook">{entry.hook}</p>
          <p className="tour-detail">{entry.detail}</p>
        </>
      ) : (
        <p className="tour-detail">Loading tour…</p>
      )}

      <div className="tour-dots">
        {examples.map((_, i) => (
          <button
            key={i}
            className={`tour-dot ${i === tourIndex ? "active" : ""}`}
            onClick={() => setTourIndex(i)}
            title={examples[i].title}
            aria-label={`Go to step ${i + 1}`}
          />
        ))}
      </div>

      <div className="tour-nav">
        <button
          className="tour-nav-btn"
          onClick={() => setTourIndex(Math.max(0, tourIndex - 1))}
          disabled={tourIndex === 0}
        >
          ← Prev
        </button>
        {tourIndex < total - 1 ? (
          <button
            className="tour-nav-btn tour-nav-btn--primary"
            onClick={() => setTourIndex(tourIndex + 1)}
          >
            Next →
          </button>
        ) : (
          <button
            className="tour-nav-btn tour-nav-btn--primary"
            onClick={endTour}
          >
            Explore freely →
          </button>
        )}
      </div>
    </div>
  );
}
