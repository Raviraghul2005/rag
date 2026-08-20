import type { VoiceStatus } from "@/hooks/useVoiceSession";

interface Props {
  status: VoiceStatus;
  onStart: () => void;
  onStop: () => void;
}

const STATUS_LABEL: Record<VoiceStatus, string> = {
  idle: "Push to talk",
  requesting_mic: "Requesting mic…",
  connecting: "Connecting…",
  listening: "Listening — tap to stop",
  processing: "Processing…",
  error: "Error — tap to retry",
};

export function MicButton({ status, onStart, onStop }: Props) {
  const isActive = status === "listening" || status === "processing";
  const isBusy = status === "requesting_mic" || status === "connecting";

  return (
    <button
      type="button"
      onClick={isActive ? onStop : onStart}
      disabled={isBusy}
      aria-pressed={isActive}
      className="flex h-24 w-24 shrink-0 items-center justify-center rounded-full border-2 transition-transform disabled:opacity-50"
      style={{
        borderColor: "var(--ink)",
        background: isActive ? "var(--rust)" : "var(--paper-100)",
        color: isActive ? "var(--paper-100)" : "var(--ink)",
        boxShadow: "var(--shadow-sm)",
      }}
      onMouseDown={(e) => (e.currentTarget.style.transform = "translateY(1px)")}
      onMouseUp={(e) => (e.currentTarget.style.transform = "translateY(0)")}
    >
      {isActive ? (
        <span className="rec-dot h-4 w-4 rounded-full" style={{ background: "var(--paper-100)" }} />
      ) : (
        <svg viewBox="0 0 24 24" fill="none" className="h-9 w-9" aria-hidden="true">
          <path
            d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <path
            d="M19 11v1a7 7 0 0 1-14 0v-1M12 19v3"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      )}
    </button>
  );
}

export { STATUS_LABEL };
