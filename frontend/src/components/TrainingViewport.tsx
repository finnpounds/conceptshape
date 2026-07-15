"use client";

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line, Billboard } from "@react-three/drei";
import * as THREE from "three";
import { useTrainingStore, type TrainingConcept } from "@/store/training";

// Same category palette as probe mode so the legend carries over.
const CATEGORY_COLORS: Record<string, string> = {
  emotions:     "#ff6b6b",
  relations:    "#69d2e7",
  abstractions: "#a8e6cf",
  states:       "#ffd700",
  nature:       "#ff8c00",
  mind:         "#b388ff",
  qualities:    "#80cbc4",
  custom:       "#e0e0e8",
};

const ANCHOR_COLORS = [
  "#ffffff", "#aaaaff", "#aaffaa", "#ffaaaa",
  "#ffff88", "#88ffff", "#ff88ff", "#ffcc88",
];

const SCALE = 4;

function lerpPos(positions: number[][], f: number): [number, number, number] {
  const i = Math.floor(f);
  const frac = f - i;
  const maxIdx = positions.length - 1;
  const a = positions[Math.min(i, maxIdx)];
  const b = positions[Math.min(i + 1, maxIdx)];
  return [
    (a[0] + (b[0] - a[0]) * frac) * SCALE,
    (a[1] + (b[1] - a[1]) * frac) * SCALE,
    (a[2] + (b[2] - a[2]) * frac) * SCALE,
  ];
}

function ConceptPoint({ concept, scrub }: { concept: TrainingConcept; scrub: number }) {
  const color = CATEGORY_COLORS[concept.category] ?? CATEGORY_COLORS.custom;
  const pos = lerpPos(concept.positions, scrub);

  return (
    <group position={pos}>
      <mesh>
        <sphereGeometry args={[0.06, 12, 12]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45} />
      </mesh>
      <Billboard>
        <Text fontSize={0.11} color={color} anchorX="center" anchorY="bottom"
          position={[0, 0.1, 0]} outlineWidth={0.018} outlineColor="#000000">
          {concept.label}
        </Text>
      </Billboard>
    </group>
  );
}

/** Drift trail: the concept's path across training so far (dim → bright). */
function ConceptTrail({ concept, scrub }: { concept: TrainingConcept; scrub: number }) {
  const color = CATEGORY_COLORS[concept.category] ?? CATEGORY_COLORS.custom;

  const segments = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= Math.floor(scrub); i++) {
      const p = concept.positions[i];
      pts.push([p[0] * SCALE, p[1] * SCALE, p[2] * SCALE]);
    }
    pts.push(lerpPos(concept.positions, scrub)); // current lerped head
    if (pts.length < 2) return [];
    return pts.slice(0, -1).map((from, i) => ({
      from,
      to: pts[i + 1],
      opacity: 0.05 + (i / Math.max(1, pts.length - 2)) * 0.3,
    }));
  }, [concept.positions, scrub]);

  return (
    <>
      {segments.map((seg, i) => (
        <Line key={i} points={[seg.from, seg.to]} color={color}
          lineWidth={1} opacity={seg.opacity} transparent />
      ))}
    </>
  );
}

function PlaybackController() {
  const { data, isPlaying, playSpeed, scrub, setScrub, setIsPlaying } = useTrainingStore();
  const ref = useRef(scrub);

  useFrame((_, delta) => {
    if (!data) return;
    if (!isPlaying) {
      ref.current = scrub;
      return;
    }
    const max = data.steps.length - 1;
    // ~0.8 checkpoint-indices per second → full 9-step sweep in ~10s at 1x
    ref.current += delta * playSpeed * 0.8;
    if (ref.current >= max) {
      ref.current = max;
      setScrub(max);
      setIsPlaying(false); // hold on the crystallized final geometry
      return;
    }
    setScrub(ref.current);
  });

  return null;
}

function TrainingScene() {
  const { data, scrub, showTrails, isLoading, error } = useTrainingStore();

  if (error) {
    return (
      <Text position={[0, 0, 0]} fontSize={0.2} color="#ff6b6b" anchorX="center" anchorY="middle">
        {error}
      </Text>
    );
  }
  if (!data || isLoading) {
    return (
      <Text position={[0, 0, 0]} fontSize={0.22} color="#555566" anchorX="center" anchorY="middle">
        Loading training time-lapse…
      </Text>
    );
  }

  return (
    <>
      <PlaybackController />

      {showTrails && data.concepts.map((c) => (
        <ConceptTrail key={`trail-${c.label}`} concept={c} scrub={scrub} />
      ))}
      {data.concepts.map((c) => (
        <ConceptPoint key={c.label} concept={c} scrub={scrub} />
      ))}

      {/* Anchor reference markers — same octahedra as probe/anchor modes */}
      {data.anchor_markers.map((marker, a) => {
        const pos = lerpPos(marker.positions, scrub);
        const color = ANCHOR_COLORS[a % ANCHOR_COLORS.length];
        return (
          <group key={marker.label} position={pos}>
            <mesh>
              <octahedronGeometry args={[0.16, 0]} />
              <meshBasicMaterial color={color} wireframe opacity={0.7} transparent />
            </mesh>
            <Billboard>
              <Text fontSize={0.13} color={color} anchorX="center" anchorY="bottom"
                position={[0, 0.24, 0]} outlineWidth={0.02} outlineColor="#000000">
                [{marker.label}]
              </Text>
            </Billboard>
          </group>
        );
      })}

      <gridHelper args={[10, 10, "#333333", "#222222"]} position={[0, -3, 0]} />
    </>
  );
}

export default function TrainingViewport() {
  const load = useTrainingStore((s) => s.load);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Canvas dpr={[1, 2]} camera={{ position: [4, 3, 4], fov: 50 }} style={{ background: "#0a0a0f" }}>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={20} />
      <TrainingScene />
    </Canvas>
  );
}
