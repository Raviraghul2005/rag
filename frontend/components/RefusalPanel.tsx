import { REFUSAL_LABELS } from "@/lib/types";
import type { PipelineResponse } from "@/lib/types";

// spec §15 item 4: the refusal must look like a designed outcome, not an error — same
// card chrome as AnswerPanel (2px ink border, paper fill), no alarm-red treatment.
// Broadside's own rule ("one accent, kept rare") argues against a semantic red/green
// status-color system anyway — the state is carried by the copy and the label, not a color.
export function RefusalPanel({ response }: { response: PipelineResponse }) {
  const reason = response.refusal_reason;
  const label = (reason && REFUSAL_LABELS[reason]) ?? "Could not answer";
  const isUnavailable = response.outcome === "unavailable";

  return (
    <div
      className="rounded-(--r-lg) p-6"
      style={{ border: "2px dashed var(--ink)", background: "var(--paper-100)" }}
    >
      <span
        className="uppercase"
        style={{
          fontFamily: "var(--font-heavy)",
          fontWeight: 800,
          fontSize: "var(--step-caption)",
          letterSpacing: ".14em",
          color: isUnavailable ? "var(--rust-deep)" : "var(--muted)",
        }}
      >
        {isUnavailable ? "Temporarily unavailable" : "Declined to answer"}
      </span>

      <p
        className="mt-4"
        style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "var(--step-statement)", lineHeight: 1.15, color: "var(--ink)" }}
      >
        {label}
      </p>

      <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2" style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}>
        {reason && <span>reason: {reason}</span>}
        {response.retrieval_top_score !== null && <span>top score: {response.retrieval_top_score.toFixed(4)}</span>}
        {response.retrieval_margin !== null && <span>margin: {response.retrieval_margin.toFixed(4)}</span>}
        {response.grounding_score !== null && <span>grounding: {(response.grounding_score * 100).toFixed(0)}%</span>}
      </div>

      {response.cited_chunks.length > 0 && (
        <div className="mt-5">
          <span
            className="uppercase"
            style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "0.7rem", letterSpacing: ".1em", color: "var(--muted)" }}
          >
            Passages found, but not used
          </span>
          <div className="mt-2 flex flex-wrap gap-2">
            {response.cited_chunks.slice(0, 5).map((chunk) => (
              <span
                key={chunk.chunk_id}
                className="rounded-(--r-sm) px-2 py-1"
                style={{ border: "1.5px solid var(--line-strong)", fontSize: "var(--step-caption)", color: "var(--muted)" }}
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
