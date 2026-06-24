"use client";

import { useCallback, useEffect, useRef } from "react";
import { useUIStore } from "@/store/ui";
import { API_BASE, HAS_LIVE_BACKEND } from "@/lib/config";

/**
 * Pings the ML backend's /health and surfaces a small status chip in the
 * sidebar. The backend (a free-tier ML Space) can be asleep — so "offline"
 * isn't an error, it's "press to wake (~30s)". Curated examples never need it.
 */
async function ping(timeoutMs: number): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

export default function BackendStatus() {
  const { backendStatus, setBackendStatus } = useUIStore();
  const wakingRef = useRef(false);

  // Initial quick check on mount.
  useEffect(() => {
    if (!HAS_LIVE_BACKEND) {
      setBackendStatus("offline");
      return;
    }
    let cancelled = false;
    setBackendStatus("checking");
    ping(6000).then((ok) => {
      if (!cancelled) setBackendStatus(ok ? "online" : "offline");
    });
    return () => {
      cancelled = true;
    };
  }, [setBackendStatus]);

  // Poll repeatedly while a cold backend wakes up.
  const wake = useCallback(async () => {
    if (wakingRef.current) return;
    wakingRef.current = true;
    setBackendStatus("checking");
    const deadline = Date.now() + 70_000; // ~70s budget for a cold start
    while (Date.now() < deadline) {
      if (await ping(8000)) {
        setBackendStatus("online");
        wakingRef.current = false;
        return;
      }
      await new Promise((r) => setTimeout(r, 3000));
    }
    setBackendStatus("offline");
    wakingRef.current = false;
  }, [setBackendStatus]);

  const meta: Record<string, { dot: string; label: string }> = {
    unknown: { dot: "#555566", label: "Backend: —" },
    checking: { dot: "#ffd700", label: "Backend: waking…" },
    online: { dot: "#3cb44b", label: "Backend: live" },
    offline: { dot: "#ff6b6b", label: "Backend: asleep" },
  };
  const m = meta[backendStatus];

  return (
    <div className="backend-status" title={`API: ${API_BASE}`}>
      <span className="backend-dot" style={{ background: m.dot }} />
      <span className="backend-label">{m.label}</span>
      {backendStatus === "offline" && HAS_LIVE_BACKEND && (
        <button className="backend-wake" onClick={wake}>
          wake (~30s)
        </button>
      )}
      {backendStatus === "checking" && (
        <span className="backend-spin" aria-hidden />
      )}
    </div>
  );
}
