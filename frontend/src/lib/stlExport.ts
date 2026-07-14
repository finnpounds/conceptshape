// Export the current token trajectories as a printable STL — each word's path
// through the model becomes a solid tube, with a bead at each end. 3D-print a
// sentence. Everything is loaded dynamically so `three` stays out of the SSR
// bundle.

import type { TokenTrajectory } from "@/store/explorer";

export async function exportTrajectoriesSTL(
  trajectories: TokenTrajectory[],
  scale: number,
  filename = "semantic-geometry.stl",
): Promise<void> {
  const THREE = await import("three");
  const { STLExporter } = await import(
    "three/examples/jsm/exporters/STLExporter.js"
  );

  const scene = new THREE.Scene();
  const bead = new THREE.SphereGeometry(0.08, 12, 12);

  for (const traj of trajectories) {
    const pts = traj.positions.map(
      (p) => new THREE.Vector3(p[0] * scale, p[1] * scale, p[2] * scale),
    );
    if (pts.length < 2) continue;

    // Smooth tube along the trajectory — solid, watertight, printable.
    const curve = new THREE.CatmullRomCurve3(pts);
    const tube = new THREE.TubeGeometry(
      curve,
      Math.max(16, pts.length * 10), // path segments
      0.04, // radius
      10, // radial segments
      false,
    );
    scene.add(new THREE.Mesh(tube));

    // Beads at the first and last layer so endpoints read clearly.
    for (const end of [pts[0], pts[pts.length - 1]]) {
      const m = new THREE.Mesh(bead);
      m.position.copy(end);
      scene.add(m);
    }
  }

  const data = new STLExporter().parse(scene, { binary: true });
  const blob = new Blob([data as unknown as BlobPart], {
    type: "application/octet-stream",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
