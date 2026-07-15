"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Line, Billboard } from "@react-three/drei";
import * as THREE from "three";
import { useSongStore } from "@/store/song";

// Color palette for the current line (same as Explore mode)
const TOKEN_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
  "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
  "#000075", "#a9a9a9",
];

const SCALE = 4.0;

function getTokenColor(i: number) {
  return TOKEN_COLORS[i % TOKEN_COLORS.length];
}

/** Desaturate a hex color by factor [0,1] for ghost rendering. */
function desaturate(hex: string, factor = 0.6): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const gray = 0.299 * r + 0.587 * g + 0.114 * b;
  const dr = Math.round(r + (gray - r) * factor);
  const dg = Math.round(g + (gray - g) * factor);
  const db = Math.round(b + (gray - b) * factor);
  return `#${dr.toString(16).padStart(2, "0")}${dg.toString(16).padStart(2, "0")}${db.toString(16).padStart(2, "0")}`;
}

// ─── Current line token ──────────────────────────────────────────────────────

function CurrentToken({
  token, positions, colorIdx, layer,
}: {
  token: string;
  positions: number[][];
  colorIdx: number;
  layer: number;
}) {
  const color = getTokenColor(colorIdx);
  const layerIdx = Math.min(Math.max(0, layer), positions.length - 1);
  const pos = positions[layerIdx];

  return (
    <group position={[pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE]}>
      {/* glow */}
      <mesh>
        <sphereGeometry args={[0.18, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.07}
          depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      {/* core */}
      <mesh>
        <sphereGeometry args={[0.07, 14, 14]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
      </mesh>
      <Billboard>
        <Text fontSize={0.13} color={color} anchorX="center" anchorY="bottom"
          position={[0, 0.13, 0]} outlineWidth={0.02} outlineColor="#000000">
          {token}
        </Text>
      </Billboard>
    </group>
  );
}

function CurrentLineTrail({
  positions, colorIdx, layer,
}: {
  positions: number[][];
  colorIdx: number;
  layer: number;
}) {
  const color = getTokenColor(colorIdx);
  const pts = useMemo(() => {
    const maxIdx = Math.min(layer + 1, positions.length);
    return positions.slice(0, maxIdx).map(
      (p) => [p[0] * SCALE, p[1] * SCALE, p[2] * SCALE] as [number, number, number]
    );
  }, [positions, layer]);

  if (pts.length < 2) return null;
  return (
    <>
      {pts.slice(0, -1).map((from, i) => (
        <Line
          key={i}
          points={[from, pts[i + 1]]}
          color={color}
          lineWidth={1.5}
          opacity={0.08 + (i / (pts.length - 2)) * 0.45}
          transparent
        />
      ))}
    </>
  );
}

// ─── Ghost lines ─────────────────────────────────────────────────────────────

function GhostLine({
  result, opacity, layer,
}: {
  result: { tokens: string[]; trajectories: { token: string; positions: number[][] }[] };
  opacity: number;
  layer: number;
}) {
  if (opacity < 0.01) return null;
  return (
    <>
      {result.trajectories.map((traj, ti) => {
        const layerIdx = Math.min(Math.max(0, layer), traj.positions.length - 1);
        const pos = traj.positions[layerIdx];
        const color = desaturate(getTokenColor(ti), 0.65);
        return (
          <mesh key={ti} position={[pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE]}>
            <sphereGeometry args={[0.045, 6, 6]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.1}
              transparent
              opacity={opacity}
            />
          </mesh>
        );
      })}
    </>
  );
}

// ─── Cumulative cloud ────────────────────────────────────────────────────────

function CumulativeCloud({
  results, layer,
}: {
  results: { tokens: string[]; trajectories: { token: string; positions: number[][] }[] }[];
  layer: number;
}) {
  // Color by line index: cool (#4363d8) → warm (#e6194b) across the song
  return (
    <>
      {results.map((result, li) => {
        const t = results.length > 1 ? li / (results.length - 1) : 0;
        const r = Math.round(67 + (230 - 67) * t);
        const g = Math.round(99 + (25 - 99) * t);
        const b = Math.round(216 + (75 - 216) * t);
        const color = `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;

        return result.trajectories.map((traj, ti) => {
          const layerIdx = Math.min(Math.max(0, layer), traj.positions.length - 1);
          const pos = traj.positions[layerIdx];
          return (
            <mesh key={`${li}-${ti}`} position={[pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE]}>
              <sphereGeometry args={[0.05, 8, 8]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.25}
                transparent opacity={0.7} />
            </mesh>
          );
        });
      })}
    </>
  );
}

// ─── Auto-orbit controller ───────────────────────────────────────────────────

function AutoOrbitController({ enabled }: { enabled: boolean }) {
  const { camera } = useThree();
  const angleRef = useRef(Math.atan2(camera.position.z, camera.position.x));
  const radiusRef = useRef(Math.sqrt(camera.position.x ** 2 + camera.position.z ** 2));

  useFrame((_, delta) => {
    if (!enabled) {
      angleRef.current = Math.atan2(camera.position.z, camera.position.x);
      radiusRef.current = Math.sqrt(camera.position.x ** 2 + camera.position.z ** 2);
      return;
    }
    angleRef.current += delta * (5 * Math.PI / 180); // 5°/s
    camera.position.x = Math.cos(angleRef.current) * radiusRef.current;
    camera.position.z = Math.sin(angleRef.current) * radiusRef.current;
    camera.lookAt(0, camera.position.y * 0.5, 0);
  });

  return null;
}

// ─── Scene ───────────────────────────────────────────────────────────────────

function SongScene({ autoOrbit }: { autoOrbit: boolean }) {
  const {
    batchResults,
    currentLineIndex,
    ghostOpacity,
    ghostDecay,
    showCumulative,
    songLayer,
    songNLayers,
  } = useSongStore();

  const resolvedLayer = songLayer === -1 ? songNLayers : songLayer;

  if (!batchResults || batchResults.length === 0) {
    return (
      <Text position={[0, 0, 0]} fontSize={0.22} color="#555566" anchorX="center" anchorY="middle">
        Pre-analyze lyrics to begin
      </Text>
    );
  }

  const currentResult = currentLineIndex >= 0 ? batchResults[currentLineIndex] : null;

  if (showCumulative) {
    return (
      <>
        <AutoOrbitController enabled={autoOrbit} />
        <CumulativeCloud results={batchResults} layer={resolvedLayer} />
        <gridHelper args={[10, 10, "#333333", "#222222"]} position={[0, -3, 0]} />
      </>
    );
  }

  // Ghost lines: all lines before current
  const ghosts = batchResults
    .slice(0, Math.max(0, currentLineIndex))
    .map((result, idx) => ({
      result,
      linesAgo: currentLineIndex - idx,
    }))
    .filter(({ linesAgo }) => ghostOpacity * Math.pow(ghostDecay, linesAgo) > 0.01);

  return (
    <>
      <AutoOrbitController enabled={autoOrbit} />

      {/* Ghosts — oldest to newest so newer ones render on top */}
      {[...ghosts].reverse().map(({ result, linesAgo }) => (
        <GhostLine
          key={result.line_index}
          result={result}
          opacity={ghostOpacity * Math.pow(ghostDecay, linesAgo)}
          layer={resolvedLayer}
        />
      ))}

      {/* Current line */}
      {currentResult?.trajectories.map((traj, ti) => (
        <group key={ti}>
          <CurrentLineTrail
            positions={traj.positions}
            colorIdx={ti}
            layer={resolvedLayer}
          />
          <CurrentToken
            token={traj.token}
            positions={traj.positions}
            colorIdx={ti}
            layer={resolvedLayer}
          />
        </group>
      ))}

      <gridHelper args={[10, 10, "#333333", "#222222"]} position={[0, -3, 0]} />
    </>
  );
}

// ─── Lyrics overlay ──────────────────────────────────────────────────────────

function LyricsOverlay() {
  const { lrcData, currentLineIndex, isPlayingAudio, showCumulative } = useSongStore();
  const line = lrcData?.lines[currentLineIndex];

  if (!line || showCumulative) return null;

  return (
    <div style={{
      position: "absolute",
      bottom: 28,
      left: "50%",
      transform: "translateX(-50%)",
      background: "rgba(0,0,0,0.52)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 20,
      padding: "6px 18px",
      fontSize: 16,
      color: "#e0e0e8",
      fontFamily: "system-ui, sans-serif",
      whiteSpace: "nowrap",
      maxWidth: "80vw",
      overflow: "hidden",
      textOverflow: "ellipsis",
      pointerEvents: "none",
      transition: "opacity 0.3s",
      opacity: isPlayingAudio || currentLineIndex >= 0 ? 1 : 0,
    }}>
      {line.text}
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export default function SongViewport() {
  const { isPlayingAudio, showCumulative } = useSongStore();
  const [autoOrbit, setAutoOrbit] = useState(false);
  const orbitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-orbit when playing or in cumulative mode
  useEffect(() => {
    setAutoOrbit(isPlayingAudio || showCumulative);
  }, [isPlayingAudio, showCumulative]);

  const handleUserInteraction = () => {
    setAutoOrbit(false);
    if (orbitTimerRef.current) clearTimeout(orbitTimerRef.current);
    // Resume orbit after 3s of no interaction — only if was playing or cumulative
    if (isPlayingAudio || showCumulative) {
      orbitTimerRef.current = setTimeout(() => setAutoOrbit(true), 3000);
    }
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [4, 3, 4], fov: 50 }}
        style={{ background: "#050508" }}
        onMouseDown={handleUserInteraction}
        onTouchStart={handleUserInteraction}
      >
        <ambientLight intensity={0.35} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={20} />
        <SongScene autoOrbit={autoOrbit} />
      </Canvas>
      <LyricsOverlay />
    </div>
  );
}
