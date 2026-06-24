"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line, Billboard, Html } from "@react-three/drei";
import * as THREE from "three";
import { useExplorerStore, type TokenTrajectory, type ProbeResult } from "@/store/explorer";

const TOKEN_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
  "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
  "#000075", "#a9a9a9",
];

// Distinct hues per model — warm red, cyan, green
const MODEL_COLORS = ["#ff6b6b", "#69d2e7", "#a8e6cf"];

const ANCHOR_COLORS = [
  "#ffffff", "#aaaaff", "#aaffaa", "#ffaaaa",
  "#ffff88", "#88ffff", "#ff88ff", "#ffcc88",
];

function getTokenColor(index: number): string {
  return TOKEN_COLORS[index % TOKEN_COLORS.length];
}

function lerpPosition(
  positions: number[][],
  layerFloat: number
): [number, number, number] {
  const layerIdx = Math.floor(layerFloat);
  const frac = layerFloat - layerIdx;
  const maxIdx = positions.length - 1;
  const a = positions[Math.min(layerIdx, maxIdx)];
  const b = positions[Math.min(layerIdx + 1, maxIdx)];
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ];
}

// ─── Absolute / Anchor shared components ────────────────────────────────────

function TokenSphere({ index }: { index: number }) {
  const viewMode = useExplorerStore((s) => s.viewMode);
  const absoluteTraj = useExplorerStore((s) => s.trajectories[index]);
  const anchorTraj = useExplorerStore((s) => s.anchorTrajectories[index]);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const spread = useExplorerStore((s) => s.spread);

  const trajectory = viewMode === "anchor" ? anchorTraj : absoluteTraj;
  const SCALE = 4 * spread;
  const color = getTokenColor(index);

  const position = useMemo(
    () => trajectory ? lerpPosition(trajectory.positions, currentLayer) : [0, 0, 0] as [number, number, number],
    [trajectory, currentLayer]
  );

  if (!trajectory) return null;

  return (
    <group position={[position[0] * SCALE, position[1] * SCALE, position[2] * SCALE]}>
      <mesh>
        <sphereGeometry args={[0.18, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.07}
          depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.07, 14, 14]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
      </mesh>
      <Billboard>
        <Text fontSize={0.13} color={color} anchorX="center" anchorY="bottom"
          position={[0, 0.13, 0]} outlineWidth={0.02} outlineColor="#000000">
          {trajectory.token}
        </Text>
      </Billboard>
    </group>
  );
}

function TrajectoryPath({ index }: { index: number }) {
  const viewMode = useExplorerStore((s) => s.viewMode);
  const absoluteTraj = useExplorerStore((s) => s.trajectories[index]);
  const anchorTraj = useExplorerStore((s) => s.anchorTrajectories[index]);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const spread = useExplorerStore((s) => s.spread);

  const trajectory = viewMode === "anchor" ? anchorTraj : absoluteTraj;
  const SCALE = 4 * spread;
  const color = getTokenColor(index);

  const segments = useMemo(() => {
    if (!trajectory) return [];
    const maxIdx = Math.ceil(currentLayer) + 1;
    const pts = trajectory.positions
      .slice(0, maxIdx)
      .map((p) => [p[0] * SCALE, p[1] * SCALE, p[2] * SCALE] as [number, number, number]);
    if (pts.length < 2) return [];
    return pts.slice(0, -1).map((from, i) => ({
      from,
      to: pts[i + 1],
      opacity: pts.length > 2 ? 0.08 + (i / (pts.length - 2)) * 0.45 : 0.45,
    }));
  }, [trajectory, currentLayer, spread]);

  return (
    <>
      {segments.map((seg, i) => (
        <Line key={i} points={[seg.from, seg.to]} color={color}
          lineWidth={1.5} opacity={seg.opacity} transparent />
      ))}
    </>
  );
}

function AttentionEdges() {
  const trajectories = useExplorerStore((s) => s.trajectories);
  const tokens = useExplorerStore((s) => s.tokens);
  const attention = useExplorerStore((s) => s.attention);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const showAttention = useExplorerStore((s) => s.showAttention);
  const threshold = useExplorerStore((s) => s.attentionThreshold);
  const hideBOS = useExplorerStore((s) => s.hideBOS);
  const spread = useExplorerStore((s) => s.spread);
  const viewMode = useExplorerStore((s) => s.viewMode);
  const SCALE = 4 * spread;

  const [hoveredEdge, setHoveredEdge] = useState<{
    fromTok: string; toTok: string; weight: number;
    pos: [number, number, number];
  } | null>(null);

  const edges = useMemo(() => {
    if (!showAttention || attention.length === 0 || viewMode !== "absolute") return [];

    const layerIdx = Math.floor(currentLayer);
    const layerEdges = attention.filter((e) => e.layer === layerIdx);
    if (layerEdges.length === 0) return [];

    const nTokens = trajectories.length;
    const avgWeights = Array.from({ length: nTokens }, () => new Array(nTokens).fill(0));
    for (const edge of layerEdges) {
      for (let i = 0; i < nTokens; i++)
        for (let j = 0; j < nTokens; j++)
          avgWeights[i][j] += edge.weights[i][j] / layerEdges.length;
    }

    // When BOS is hidden, renormalize each row over non-BOS tokens so that
    // BOS-sink attention doesn't push all remaining weights below threshold.
    // This reveals the true structure of how semantic tokens attend to each other.
    if (hideBOS) {
      for (let i = 1; i < nTokens; i++) {
        let rowSum = 0;
        for (let j = 1; j < nTokens; j++) {
          if (j !== i) rowSum += avgWeights[i][j];
        }
        if (rowSum > 1e-8) {
          for (let j = 1; j < nTokens; j++) {
            if (j !== i) avgWeights[i][j] /= rowSum;
          }
        }
      }
    }

    const result: {
      from: [number, number, number]; to: [number, number, number];
      mid: [number, number, number]; weight: number;
      fromTok: string; toTok: string;
    }[] = [];

    for (let i = 0; i < nTokens; i++) {
      if (hideBOS && i === 0) continue;
      for (let j = 0; j < nTokens; j++) {
        if (i === j || (hideBOS && j === 0)) continue;
        const w = avgWeights[i][j];
        if (w < threshold) continue;
        const fp = lerpPosition(trajectories[i].positions, currentLayer);
        const tp = lerpPosition(trajectories[j].positions, currentLayer);
        result.push({
          from: [fp[0] * SCALE, fp[1] * SCALE, fp[2] * SCALE],
          to:   [tp[0] * SCALE, tp[1] * SCALE, tp[2] * SCALE],
          mid:  [((fp[0]+tp[0])/2)*SCALE, ((fp[1]+tp[1])/2)*SCALE, ((fp[2]+tp[2])/2)*SCALE],
          weight: w,
          fromTok: tokens[i] ?? `t${i}`,
          toTok:   tokens[j] ?? `t${j}`,
        });
      }
    }
    return result;
  }, [trajectories, tokens, attention, currentLayer, showAttention, threshold, hideBOS, spread, viewMode]);

  return (
    <>
      {edges.map((edge, i) => {
        const v = 0.2 + edge.weight * 0.8;
        return (
          <group key={i}>
            <Line
              points={[edge.from, edge.to]}
              color={new THREE.Color(v, v, v)}
              lineWidth={Math.max(0.5, edge.weight * 4)}
              opacity={edge.weight * 0.6 + 0.1}
              transparent
              onPointerEnter={() => setHoveredEdge({
                fromTok: edge.fromTok, toTok: edge.toTok,
                weight: edge.weight, pos: edge.mid,
              })}
              onPointerLeave={() => setHoveredEdge(null)}
            />
          </group>
        );
      })}
      {hoveredEdge && (
        <Html position={hoveredEdge.pos} style={{ pointerEvents: "none" }}>
          <div style={{
            background: "rgba(10,10,15,0.92)", border: "1px solid #2a2a3a",
            borderRadius: "4px", padding: "4px 8px", fontSize: "11px",
            color: "#e0e0e8", fontFamily: "monospace", whiteSpace: "nowrap", lineHeight: "1.5",
          }}>
            <span style={{ color: "#888899" }}>{hoveredEdge.fromTok}</span>{" → "}
            <span style={{ color: "#888899" }}>{hoveredEdge.toTok}</span><br />
            <span style={{ color: "#555566" }}>weight: </span>{hoveredEdge.weight.toFixed(3)}
          </div>
        </Html>
      )}
    </>
  );
}

function AnchorMarkers() {
  const anchorMarkers = useExplorerStore((s) => s.anchorMarkers);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const spread = useExplorerStore((s) => s.spread);
  const viewMode = useExplorerStore((s) => s.viewMode);
  const SCALE = 4 * spread;

  if (viewMode !== "anchor" || anchorMarkers.length === 0) return null;

  return (
    <>
      {anchorMarkers.map((marker, a) => {
        const pos = lerpPosition(marker.positions, currentLayer);
        const color = ANCHOR_COLORS[a % ANCHOR_COLORS.length];
        return (
          <group key={a} position={[pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE]}>
            <mesh>
              <octahedronGeometry args={[0.18, 0]} />
              <meshBasicMaterial color={color} wireframe opacity={0.7} transparent />
            </mesh>
            <mesh>
              <octahedronGeometry args={[0.32, 0]} />
              <meshBasicMaterial color={color} transparent opacity={0.04}
                depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>
            <Billboard>
              <Text fontSize={0.14} color={color} anchorX="center" anchorY="bottom"
                position={[0, 0.28, 0]} outlineWidth={0.02} outlineColor="#000000">
                [{marker.label}]
              </Text>
            </Billboard>
          </group>
        );
      })}
    </>
  );
}

// ─── Compare scene components ────────────────────────────────────────────────

function CompareTokenSphere({
  trajectory, modelColor, nLayers,
}: { trajectory: TokenTrajectory; modelColor: string; nLayers: number }) {
  const compareLayer = useExplorerStore((s) => s.compareLayer);
  const spread = useExplorerStore((s) => s.spread);
  const SCALE = 4 * spread;
  const layerFloat = compareLayer * nLayers;

  const position = useMemo(
    () => lerpPosition(trajectory.positions, layerFloat),
    [trajectory.positions, layerFloat]
  );

  return (
    <group position={[position[0] * SCALE, position[1] * SCALE, position[2] * SCALE]}>
      <mesh>
        <sphereGeometry args={[0.07, 12, 12]} />
        <meshStandardMaterial color={modelColor} emissive={modelColor} emissiveIntensity={0.35} />
      </mesh>
      <Billboard>
        <Text fontSize={0.11} color={modelColor} anchorX="center" anchorY="bottom"
          position={[0, 0.12, 0]} outlineWidth={0.02} outlineColor="#000000">
          {trajectory.token}
        </Text>
      </Billboard>
    </group>
  );
}

function CompareTrajectoryPath({
  trajectory, modelColor, nLayers,
}: { trajectory: TokenTrajectory; modelColor: string; nLayers: number }) {
  const compareLayer = useExplorerStore((s) => s.compareLayer);
  const spread = useExplorerStore((s) => s.spread);
  const SCALE = 4 * spread;
  const layerFloat = compareLayer * nLayers;

  const segments = useMemo(() => {
    const maxIdx = Math.ceil(layerFloat) + 1;
    const pts = trajectory.positions
      .slice(0, maxIdx)
      .map((p) => [p[0] * SCALE, p[1] * SCALE, p[2] * SCALE] as [number, number, number]);
    if (pts.length < 2) return [];
    return pts.slice(0, -1).map((from, i) => ({
      from,
      to: pts[i + 1],
      opacity: pts.length > 2 ? 0.06 + (i / (pts.length - 2)) * 0.3 : 0.3,
    }));
  }, [trajectory.positions, layerFloat, spread]);

  return (
    <>
      {segments.map((seg, i) => (
        <Line key={i} points={[seg.from, seg.to]} color={modelColor}
          lineWidth={1} opacity={seg.opacity} transparent />
      ))}
    </>
  );
}

/** Renders all models' trajectories in the shared anchor-relative space. */
function CompareScene() {
  const compareData = useExplorerStore((s) => s.compareData);
  const hideBOS = useExplorerStore((s) => s.hideBOS);

  return (
    <>
      {compareData.map((model, mi) => {
        const modelColor = MODEL_COLORS[mi % MODEL_COLORS.length];
        const visibleIndices = model.trajectories
          .map((_, ti) => ti)
          .filter((ti) => !(hideBOS && ti === 0));

        return visibleIndices.map((ti) => (
          <group key={`${mi}-${ti}`}>
            <CompareTrajectoryPath
              trajectory={model.trajectories[ti]}
              modelColor={modelColor}
              nLayers={model.nLayers}
            />
            <CompareTokenSphere
              trajectory={model.trajectories[ti]}
              modelColor={modelColor}
              nLayers={model.nLayers}
            />
          </group>
        ));
      })}
    </>
  );
}

// ─── Probe scene components ──────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  emotions:    "#ff6b6b",
  relations:   "#69d2e7",
  abstractions:"#a8e6cf",
  states:      "#ffd700",
  nature:      "#ff8c00",
  mind:        "#b388ff",
  qualities:   "#80cbc4",
  custom:      "#e0e0e8",
};

function ProbeConceptSphere({ probe }: { probe: ProbeResult }) {
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const spread = useExplorerStore((s) => s.spread);
  const SCALE = 4 * spread;
  const [hovered, setHovered] = useState(false);

  const color = CATEGORY_COLORS[probe.category] ?? CATEGORY_COLORS.custom;

  const position = useMemo(
    () => lerpPosition(probe.positions, currentLayer),
    [probe.positions, currentLayer]
  );

  const scaled: [number, number, number] = [
    position[0] * SCALE,
    position[1] * SCALE,
    position[2] * SCALE,
  ];

  // Uncertainty [0,1]: high = far from all anchors = uncertain = larger halo
  const glowOpacity = 0.04 + probe.uncertainty * 0.18;
  const glowRadius = 0.12 + probe.uncertainty * 0.18;

  return (
    <group position={scaled}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}>
      {/* uncertainty halo */}
      <mesh>
        <sphereGeometry args={[glowRadius, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={glowOpacity}
          depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      {/* core sphere */}
      <mesh>
        <sphereGeometry args={[0.05, 10, 10]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} />
      </mesh>
      {hovered && (
        <Html style={{ pointerEvents: "none" }}>
          <div style={{
            background: "rgba(10,10,15,0.92)", border: "1px solid #2a2a3a",
            borderRadius: "4px", padding: "3px 7px", fontSize: "11px",
            color: "#e0e0e8", fontFamily: "monospace", whiteSpace: "nowrap",
            transform: "translate(8px, -50%)",
          }}>
            <span style={{ color }}>{probe.label}</span>
            <span style={{ color: "#555566", marginLeft: 6 }}>{probe.category}</span>
            <br />
            <span style={{ color: "#555566" }}>uncertainty: </span>
            <span style={{ color: "#888899" }}>{probe.uncertainty.toFixed(2)}</span>
          </div>
        </Html>
      )}
    </group>
  );
}

function ProbeScene() {
  const probeResults = useExplorerStore((s) => s.probeResults);
  const probeAnchorMarkers = useExplorerStore((s) => s.probeAnchorMarkers);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const spread = useExplorerStore((s) => s.spread);
  const SCALE = 4 * spread;

  return (
    <>
      {probeResults.map((probe, i) => (
        <ProbeConceptSphere key={i} probe={probe} />
      ))}
      {/* Anchor markers — same octahedra style as anchor mode */}
      {probeAnchorMarkers.map((marker, a) => {
        const pos = lerpPosition(marker.positions, currentLayer);
        const color = ANCHOR_COLORS[a % ANCHOR_COLORS.length];
        return (
          <group key={a} position={[pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE]}>
            <mesh>
              <octahedronGeometry args={[0.18, 0]} />
              <meshBasicMaterial color={color} wireframe opacity={0.7} transparent />
            </mesh>
            <mesh>
              <octahedronGeometry args={[0.32, 0]} />
              <meshBasicMaterial color={color} transparent opacity={0.04}
                depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>
            <Billboard>
              <Text fontSize={0.14} color={color} anchorX="center" anchorY="bottom"
                position={[0, 0.28, 0]} outlineWidth={0.02} outlineColor="#000000">
                [{marker.label}]
              </Text>
            </Billboard>
          </group>
        );
      })}
    </>
  );
}

// ─── Playback controllers ────────────────────────────────────────────────────

function AbsolutePlaybackController() {
  const isPlaying = useExplorerStore((s) => s.isPlaying);
  const playSpeed = useExplorerStore((s) => s.playSpeed);
  const nLayers = useExplorerStore((s) => s.nLayers);
  const probeNLayers = useExplorerStore((s) => s.probeNLayers);
  const viewMode = useExplorerStore((s) => s.viewMode);
  const currentLayer = useExplorerStore((s) => s.currentLayer);
  const setCurrentLayer = useExplorerStore((s) => s.setCurrentLayer);
  const layerRef = useRef(0);

  useFrame((_, delta) => {
    const maxLayer = viewMode === "probe" ? probeNLayers : nLayers;
    if (!isPlaying || maxLayer === 0) {
      layerRef.current = currentLayer;
      return;
    }
    layerRef.current += delta * playSpeed;
    if (layerRef.current > maxLayer) layerRef.current = 0;
    setCurrentLayer(layerRef.current);
  });

  return null;
}

function ComparePlaybackController() {
  const isPlaying = useExplorerStore((s) => s.isPlaying);
  const playSpeed = useExplorerStore((s) => s.playSpeed);
  const compareLayer = useExplorerStore((s) => s.compareLayer);
  const setCompareLayer = useExplorerStore((s) => s.setCompareLayer);
  const layerRef = useRef(0);

  useFrame((_, delta) => {
    if (!isPlaying) {
      layerRef.current = compareLayer;
      return;
    }
    // 0-1 range, playSpeed=1 takes ~8s end-to-end at 60fps
    layerRef.current += delta * playSpeed * 0.125;
    if (layerRef.current > 1) layerRef.current = 0;
    setCompareLayer(layerRef.current);
  });

  return null;
}

// ─── Scene helpers ───────────────────────────────────────────────────────────

function ReferenceGrid() {
  return <gridHelper args={[10, 10, "#333333", "#222222"]} position={[0, -3, 0]} />;
}

function AxisLabels() {
  const viewMode = useExplorerStore((s) => s.viewMode);
  const explainedVariance = useExplorerStore((s) => s.explainedVariance);
  const anchorEV = useExplorerStore((s) => s.anchorExplainedVariance);
  const compareEV = useExplorerStore((s) => s.compareExplainedVariance);

  const probeEV = useExplorerStore((s) => s.probeExplainedVariance);

  const ev =
    viewMode === "anchor"  ? anchorEV :
    viewMode === "compare" ? compareEV :
    viewMode === "probe"   ? probeEV :
    explainedVariance;

  if (ev.length < 3) return null;

  const pct = ev.map((v) => (v * 100).toFixed(1));
  const prefix =
    viewMode === "anchor"  ? "Anc-PC" :
    viewMode === "compare" ? "Cmp-PC" :
    viewMode === "probe"   ? "Prb-PC" :
    "PC";

  return (
    <>
      <Text position={[5.5, -3, 0]} fontSize={0.18} color="#444455" anchorX="left" anchorY="middle">
        {`${prefix}1 (${pct[0]}%)`}
      </Text>
      <Text position={[0, 2.2, 0]} fontSize={0.18} color="#444455" anchorX="center" anchorY="bottom">
        {`${prefix}2 (${pct[1]}%)`}
      </Text>
      <Text position={[0, -3, 5.5]} fontSize={0.18} color="#444455" anchorX="left" anchorY="middle">
        {`${prefix}3 (${pct[2]}%)`}
      </Text>
    </>
  );
}

// ─── Main viewport ───────────────────────────────────────────────────────────

export default function Viewport() {
  const viewMode = useExplorerStore((s) => s.viewMode);
  const trajectories = useExplorerStore((s) => s.trajectories);
  const anchorTrajectories = useExplorerStore((s) => s.anchorTrajectories);
  const compareData = useExplorerStore((s) => s.compareData);
  const hideBOS = useExplorerStore((s) => s.hideBOS);

  const activeTrajectories =
    viewMode === "anchor"  ? anchorTrajectories :
    viewMode === "compare" ? [] : // CompareScene handles its own rendering
    viewMode === "probe"   ? [] : // ProbeScene handles its own rendering
    trajectories;

  const visibleIndices = activeTrajectories
    .map((_, i) => i)
    .filter((i) => !(hideBOS && i === 0));

  return (
    <Canvas camera={{ position: [4, 3, 4], fov: 50 }} style={{ background: "#0a0a0f" }}>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />

      <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={20} />

      <ReferenceGrid />
      <AxisLabels />

      {/* Mode-specific playback and content */}
      {viewMode === "compare" ? (
        <>
          <ComparePlaybackController />
          {compareData.length > 0 && <CompareScene />}
        </>
      ) : viewMode === "probe" ? (
        <>
          <AbsolutePlaybackController />
          <ProbeScene />
        </>
      ) : (
        <>
          <AbsolutePlaybackController />
          <AttentionEdges />
          <AnchorMarkers />
          {visibleIndices.map((i) => (
            <group key={i}>
              <TrajectoryPath index={i} />
              <TokenSphere index={i} />
            </group>
          ))}
        </>
      )}
    </Canvas>
  );
}
