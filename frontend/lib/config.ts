// The only public env var this frontend needs (spec §15/§16.1) — the FastAPI Space
// URL. Never SARVAM_API_KEY or any provider key: those stay server-side only.
const RAW_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:7860";

export const API_BASE_URL = RAW_BASE_URL.replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export function wsUrl(path: string): string {
  const wsBase = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBase}${path.startsWith("/") ? path : `/${path}`}`;
}

export const LATENCY_BUDGET_MS = 200;
