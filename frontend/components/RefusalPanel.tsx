import { REFUSAL_LABELS } from "@/lib/types";
import type { PipelineResponse } from "@/lib/types";

// spec §15 item 4: the refusal must look like a designed outcome, not an error — same
// panel chrome as AnswerPanel, an accent color instead of a generic red "failure"
// treatment, and the actual score that triggered the decision shown plainly.
export function RefusalPanel({ response }: { response: PipelineResponse }) {
  const reason = response.refusal_reason;
  const label = (reason && REFUSAL_LABELS[reason]) ?? "Could not answer";
  const isUnavailable = response.outcome === "unavailable";

  return (
    <div
      className={`rounded-lg border p-5 ${
        isUnavailable ? "border-warn/30" : "border-border-strong"
      }`}
    >
      <span
        className={`font-mono text-xs uppercase tracking-wider ${
          isUnavailable ? "text-warn" : "text-text-dim"
        }`}
      >
        {isUnavailable ? "temporarily unavailable" : "declined to answer"}
      </span>

      <p className="mt-3 text-lg leading-relaxed text-text">{label}</p>

      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 font-mono text-xs text-text-faint">
        {reason && <span>reason: {reason}</span>}
        {response.retrieval_top_score !== null && (
          <span>top score: {response.retrieval_top_score.toFixed(4)}</span>
        )}
        {response.retrieval_margin !== null && (
          <span>margin: {response.retrieval_margin.toFixed(4)}</span>
        )}
        {response.grounding_score !== null && (
          <span>grounding: {(response.grounding_score * 100).toFixed(0)}%</span>
        )}
      </div>

      {response.cited_chunks.length > 0 && (
        <div className="mt-4">
          <span className="font-mono text-[11px] uppercase tracking-wider text-text-faint">
            passages found, but not used
          </span>
          <div className="mt-2 flex flex-wrap gap-2">
            {response.cited_chunks.slice(0, 5).map((chunk) => (
              <span
                key={chunk.chunk_id}
                className="rounded border border-border bg-bg-panel-raised px-2 py-1 font-mono text-xs text-text-faint"
              >
                [{chunk.chunk_id}]
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
