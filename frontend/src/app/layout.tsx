import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000";

const TITLE = "Semantic Geometry Explorer";
const DESCRIPTION =
  "An interactive 3D microscope for the inside of a language model — watch meaning take physical shape through transformer layers.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  keywords: [
    "interpretability",
    "transformers",
    "mechanistic interpretability",
    "representation engineering",
    "3D visualization",
    "PCA",
    "UMAP",
    "persistent homology",
    "TransformerLens",
  ],
  authors: [{ name: "Finn" }],
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: SITE_URL,
    siteName: TITLE,
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
