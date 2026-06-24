"use client";

import { useCallback, useEffect, useRef } from "react";
import { useSongStore } from "@/store/song";
import { parseLrc, findCurrentLineIndex } from "@/lib/lrc-parser";
import { analyzeBatch } from "@/lib/api";

export default function SongControls() {
  const {
    lrcData, setLrcData,
    audioUrl, audioFilename, setAudioUrl,
    batchResults, setBatchResults,
    songNLayers, songModelName,
    isPlayingAudio, setIsPlayingAudio,
    audioCurrentTime, setAudioCurrentTime,
    audioDuration, setAudioDuration,
    currentLineIndex, setCurrentLineIndex,
    ghostOpacity, setGhostOpacity,
    ghostDecay, setGhostDecay,
    showCumulative, setShowCumulative,
    songLayer, setSongLayer,
    songIsPreprocessing, setSongIsPreprocessing,
    reset,
  } = useSongStore();

  const audioRef = useRef<HTMLAudioElement>(null);
  const lrcInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Keep audio element in sync with store's play state
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (isPlayingAudio) el.play().catch(() => setIsPlayingAudio(false));
    else el.pause();
  }, [isPlayingAudio]);

  // Audio element event handlers
  const onTimeUpdate = useCallback(() => {
    const el = audioRef.current;
    if (!el || !lrcData) return;
    const t = el.currentTime;
    setAudioCurrentTime(t);
    setCurrentLineIndex(findCurrentLineIndex(lrcData.lines, t));
  }, [lrcData]);

  const onEnded = useCallback(() => {
    setIsPlayingAudio(false);
    setShowCumulative(true); // reveal full shape on song end
  }, []);

  // --- File loaders ---
  const handleLrcFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (content) setLrcData(parseLrc(content));
    };
    reader.readAsText(file);
  };

  const handleAudioFile = (file: File) => {
    const prev = audioUrl;
    if (prev) URL.revokeObjectURL(prev);
    const url = URL.createObjectURL(file);
    setAudioUrl(url, file.name);
    // Reset audio element src
    const el = audioRef.current;
    if (el) { el.src = url; el.load(); }
  };

  const handleLrcDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = Array.from(e.dataTransfer.files).find((f) => f.name.endsWith(".lrc"));
    if (file) handleLrcFile(file);
  };

  const handleAudioDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = Array.from(e.dataTransfer.files).find((f) =>
      /\.(mp3|wav|ogg|m4a|flac|aac)$/i.test(f.name)
    );
    if (file) handleAudioFile(file);
  };

  // --- Pre-analyze ---
  const handlePreanalyze = useCallback(async () => {
    if (!lrcData) return;
    const lines = lrcData.lines.map((l) => l.text).filter(Boolean);
    if (lines.length === 0) return;
    setSongIsPreprocessing(true);
    try {
      const data = await analyzeBatch(lines);
      setBatchResults(data.results, data.n_layers, data.explained_variance, data.model_name);
    } catch (err) {
      console.error("Batch analysis failed:", err);
    } finally {
      setSongIsPreprocessing(false);
    }
  }, [lrcData]);

  // --- Playback ---
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    const el = audioRef.current;
    if (el) el.currentTime = t;
    setAudioCurrentTime(t);
    if (lrcData) setCurrentLineIndex(findCurrentLineIndex(lrcData.lines, t));
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const currentLine = lrcData?.lines[currentLineIndex] ?? null;
  const isReady = !!batchResults && !songIsPreprocessing;
  const hasLrc = !!lrcData && lrcData.lines.length > 0;
  const hasAudio = !!audioUrl;

  const layerDisplay = songLayer === -1
    ? `Last (${songNLayers})`
    : `${songLayer}`;

  return (
    <div className="controls-panel">
      {/* LRC file */}
      <div className="control-section">
        <label className="control-label">Lyrics File (.lrc)</label>
        <div
          className={`song-dropzone ${hasLrc ? "song-dropzone--loaded" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleLrcDrop}
          onClick={() => lrcInputRef.current?.click()}
        >
          {hasLrc
            ? `✓ ${lrcData!.metadata.title ?? "lyrics"} — ${lrcData!.lines.length} lines`
            : "Drop .lrc file or click"}
        </div>
        <input
          ref={lrcInputRef}
          type="file"
          accept=".lrc"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleLrcFile(f); e.target.value = ""; }}
        />
      </div>

      {/* Audio file */}
      <div className="control-section">
        <label className="control-label">Audio File</label>
        <div
          className={`song-dropzone ${hasAudio ? "song-dropzone--loaded" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleAudioDrop}
          onClick={() => audioInputRef.current?.click()}
        >
          {hasAudio ? `✓ ${audioFilename}` : "Drop audio or click (mp3, wav, ogg…)"}
        </div>
        <input
          ref={audioInputRef}
          type="file"
          accept="audio/*"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleAudioFile(f); e.target.value = ""; }}
        />
        {/* Hidden audio element */}
        <audio
          ref={audioRef}
          onTimeUpdate={onTimeUpdate}
          onDurationChange={(e) => setAudioDuration((e.target as HTMLAudioElement).duration)}
          onEnded={onEnded}
          preload="metadata"
          style={{ display: "none" }}
        />
      </div>

      {/* Status & pre-analyze */}
      <div className="control-section">
        <label className="control-label">Status</label>
        {!hasLrc && <p className="info-value" style={{ color: "var(--text-dim)" }}>Load an .lrc file</p>}
        {hasLrc && !isReady && !songIsPreprocessing && (
          <p className="info-value" style={{ color: "var(--text-dim)", marginBottom: 6 }}>
            {lrcData!.lines.length} lines ready to analyze
          </p>
        )}
        {songIsPreprocessing && (
          <p className="info-value" style={{ color: "var(--accent)", marginBottom: 6 }}>
            Embedding {lrcData?.lines.length} lines…
          </p>
        )}
        {isReady && (
          <div className="info-row">
            <span className="info-label">Ready</span>
            <span className="info-value" style={{ color: "#a8e6cf" }}>
              ✓ {batchResults!.length} lines · {songModelName}
            </span>
          </div>
        )}
        <button
          className="anchor-compute-btn"
          style={{ marginTop: 6, borderColor: "#b388ff", color: "#b388ff", background: "rgba(179,136,255,0.1)" }}
          onClick={handlePreanalyze}
          disabled={!hasLrc || songIsPreprocessing}
        >
          {songIsPreprocessing ? "Pre-analyzing…" : "Pre-analyze Lyrics"}
        </button>
      </div>

      {/* Playback */}
      {hasAudio && (
        <div className="control-section">
          <div className="slider-header">
            <label className="control-label">
              {formatTime(audioCurrentTime)} / {formatTime(audioDuration)}
            </label>
            <div className="playback-controls">
              <button
                className="playback-btn"
                onClick={() => {
                  const el = audioRef.current;
                  if (el) { el.currentTime = 0; }
                  setAudioCurrentTime(0);
                  setCurrentLineIndex(-1);
                  setIsPlayingAudio(false);
                }}
                title="Reset"
              >⟲</button>
              <button
                className="playback-btn"
                onClick={() => setIsPlayingAudio(!isPlayingAudio)}
                disabled={!isReady}
              >
                {isPlayingAudio ? "⏸" : "▶"}
              </button>
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={audioDuration || 1}
            step={0.1}
            value={audioCurrentTime}
            onChange={handleSeek}
            className="layer-slider"
          />
          {currentLine && (
            <div className="song-current-line">
              ♪ {currentLine.text}
            </div>
          )}
        </div>
      )}

      {/* Ghost trails */}
      <div className="control-section">
        <label className="control-label">Ghost Trails</label>
        <div className="speed-row">
          <label className="info-label">Opacity</label>
          <input type="range" min={0.05} max={0.8} step={0.05} value={ghostOpacity}
            onChange={(e) => setGhostOpacity(parseFloat(e.target.value))} className="speed-slider" />
          <span className="info-value">{ghostOpacity.toFixed(2)}</span>
        </div>
        <div className="speed-row">
          <label className="info-label">Decay</label>
          <input type="range" min={0.5} max={0.98} step={0.02} value={ghostDecay}
            onChange={(e) => setGhostDecay(parseFloat(e.target.value))} className="speed-slider" />
          <span className="info-value">{ghostDecay.toFixed(2)}</span>
        </div>
      </div>

      {/* View */}
      <div className="control-section">
        <label className="control-label">View</label>
        <div className="method-toggle">
          <button
            className={`method-btn ${!showCumulative ? "active" : ""}`}
            onClick={() => setShowCumulative(false)}
            title="Show current + fading ghosts"
          >
            Current
          </button>
          <button
            className={`method-btn ${showCumulative ? "active" : ""}`}
            onClick={() => setShowCumulative(true)}
            title="All lines at once — shape of the song"
          >
            Cumulative
          </button>
        </div>
      </div>

      {/* Layer */}
      {songNLayers > 0 && (
        <div className="control-section">
          <div className="info-row">
            <span className="info-label">Layer</span>
            <span className="info-value">{layerDisplay}</span>
          </div>
          <input
            type="range"
            min={-1}
            max={songNLayers}
            step={1}
            value={songLayer}
            onChange={(e) => setSongLayer(parseInt(e.target.value))}
            className="layer-slider"
          />
        </div>
      )}

      {/* Reset */}
      <div className="control-section">
        <button className="anchor-remove-btn"
          style={{ width: "100%", fontSize: 10, height: "auto", padding: "5px" }}
          onClick={reset}>
          Reset Song
        </button>
      </div>
    </div>
  );
}
