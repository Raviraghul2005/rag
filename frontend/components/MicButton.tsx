import type { VoiceStatus } from "@/hooks/useVoiceSession";

interface Props {
  status: VoiceStatus;
  onStart: () => void;
  onStop: () => void;
}

const STATUS_LABEL: Record<VoiceStatus, string> = {
  idle: "push to talk",
  requesting_mic: "requesting mic…",
  connecting: "connecting…",
  listening: "listening — tap to stop",
  processing: "processing…",
  error: "error — tap to retry",
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
      className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-full border-2 transition-colors disabled:opacity-50 ${
        isActive
          ? "border-danger bg-danger/10 text-danger"
          : "border-accent bg-accent/10 text-accent hover:bg-accent/20"
      }`}
    >
      {isActive ? (
        <span className="rec-dot h-4 w-4 rounded-full bg-danger" />
      ) : (
        <svg viewBox="0 0 24 24" fill="none" className="h-8 w-8" aria-hidden="true">
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
