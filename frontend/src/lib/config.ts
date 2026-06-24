// Central place for deploy-time configuration.
// These are read at build time from NEXT_PUBLIC_* env vars (set in Vercel),
// with sensible local-dev fallbacks.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Link shown in the About panel + sidebar. Update once the repo is public.
export const GITHUB_URL =
  process.env.NEXT_PUBLIC_GITHUB_URL ||
  "https://github.com/your-username/semantic-geometry-explorer";

// Whether a live backend is expected to exist for custom input. When false
// (e.g. a gallery-only deploy), the UI nudges visitors toward the tour.
export const HAS_LIVE_BACKEND =
  (process.env.NEXT_PUBLIC_HAS_LIVE_BACKEND ?? "true") !== "false";
