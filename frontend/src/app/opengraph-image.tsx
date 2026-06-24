import { ImageResponse } from "next/og";

// Branded social-share card, generated at build/request time (no binary asset).
export const runtime = "edge";
export const alt = "Semantic Geometry Explorer";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "80px",
          backgroundColor: "#0a0a0f",
          backgroundImage:
            "radial-gradient(circle at 25% 30%, #1b2350 0%, transparent 45%), radial-gradient(circle at 78% 72%, #2a1c4a 0%, transparent 48%)",
          fontFamily: "monospace",
        }}
      >
        <div
          style={{
            color: "#8ba4ff",
            fontSize: 26,
            letterSpacing: 6,
            textTransform: "uppercase",
            marginBottom: 28,
          }}
        >
          Interpretability · 3D · Transformers
        </div>
        <div
          style={{
            color: "#e8e8f0",
            fontSize: 82,
            fontWeight: 700,
            lineHeight: 1.05,
            letterSpacing: -2,
          }}
        >
          Semantic Geometry
        </div>
        <div
          style={{
            color: "#b388ff",
            fontSize: 82,
            fontWeight: 700,
            lineHeight: 1.05,
            letterSpacing: -2,
            marginBottom: 36,
          }}
        >
          Explorer
        </div>
        <div style={{ color: "#9a9ab0", fontSize: 32, lineHeight: 1.4, maxWidth: 900 }}>
          Watch meaning take physical shape inside a language model — token by
          token, layer by layer, in three dimensions.
        </div>
      </div>
    ),
    { ...size },
  );
}
