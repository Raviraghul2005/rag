// The only public env var this frontend needs (spec §15/§16.1) — the FastAPI Space
// URL. Never SARVAM_API_KEY or any provider key: those stay server-side only.
const RAW_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:7860";

// A bare domain with no scheme (e.g. "rag-production-997b.up.railway.app" instead of
// "https://rag-production-997b.up.railway.app") is a real misconfiguration that's easy
// to make when pasting a Railway/Vercel domain — and it fails silently in a confusing
// way: the browser treats a scheme-less string as a *path relative to the current
// page*, not a different host, so every fetch/WebSocket call quietly resolves against
// the frontend's own origin instead of the backend and 404s. Defend against it here
// rather than trusting the env var's format.
const WITH_SCHEME = /^https?:\/\//.test(RAW_BASE_URL) ? RAW_BASE_URL : `https://${RAW_BASE_URL}`;

export const API_BASE_URL = WITH_SCHEME.replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export function wsUrl(path: string): string {
  const wsBase = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBase}${path.startsWith("/") ? path : `/${path}`}`;
}

export const LATENCY_BUDGET_MS = 200;
