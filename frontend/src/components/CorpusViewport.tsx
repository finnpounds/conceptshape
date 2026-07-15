"use client";

import { useRef, useMemo, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { useCorpusStore } from "@/store/corpus";
import { CORPUS_COLORS } from "./CorpusControls";

// ─── Point cloud ─────────────────────────────────────────────────────────────

interface PointCloudProps {
  points: number[][];
  color: string;
  spread: number;
}

function PointCloud({ points, color, spread }: PointCloudProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current) return;
    const SCALE = 4 * spread;
    points.forEach(([x, y, z], i) => {
      dummy.position.set(x * SCALE, y * SCALE, z * SCALE);
      dummy.scale.setScalar(0.04);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [points, spread, dummy]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, points.length]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.2}
        transparent
        opacity={0.65}
      />
    </instancedMesh>
  );
}

// ─── Label cloud centroid ────────────────────────────────────────────────────

function CloudLabel({ points, label, color, spread }: PointCloudProps & { label: string }) {
  const centroid = useMemo(() => {
    if (points.length === 0) return [0, 0, 0] as [number, number, number];
    const SCALE = 4 * spread;
    const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0]);
    return [
      (sum[0] / points.length) * SCALE,
      (sum[1] / points.length) * SCALE + 0.5,
      (sum[2] / points.length) * SCALE,
    ] as [number, number, number];
  }, [points, spread]);

  return (
    <Text
      position={centroid}
      fontSize={0.18}
      color={color}
      anchorX="center"
      anchorY="bottom"
      outlineWidth={0.02}
      outlineColor="#000000"
    >
      {label}
    </Text>
  );
}

// ─── Scene ───────────────────────────────────────────────────────────────────

function CorpusScene() {
  const { corpusResults, visibleShapes } = useCorpusStore();
  const spread = 1.0; // fixed for corpus mode

  if (!corpusResults || corpusResults.shapes.length === 0) {
    return (
      <Text position={[0, 0, 0]} fontSize={0.22} color="#555566" anchorX="center" anchorY="middle">
        Add texts and click Analyze
      </Text>
    );
  }

  return (
    <>
      {corpusResults.shapes.map((shape, i) => {
        if (!visibleShapes.has(i)) return null;
        const color = CORPUS_COLORS[i % CORPUS_COLORS.length];
        return (
          <group key={i}>
            <PointCloud points={shape.pointCloud3d} color={color} spread={spread} />
            <CloudLabel
              points={shape.pointCloud3d}
              label={shape.label}
              color={color}
              spread={spread}
            />
          </group>
        );
      })}
      <gridHelper args={[10, 10, "#333333", "#222222"]} position={[0, -3, 0]} />
    </>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

export default function CorpusViewport() {
  return (
    <Canvas dpr={[1, 2]} camera={{ position: [4, 3, 4], fov: 50 }} style={{ background: "#050508" }}>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={20} />
      <CorpusScene />
    </Canvas>
  );
}
