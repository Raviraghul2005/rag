import { LATENCY_BUDGET_MS } from "@/lib/config";
import type { StageTimings } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  input_guardrails: "Guardrails",
  retrieve: "Retrieve",
  retrieval_gate: "Confidence gate",
  generate: "Generate",
  grounding: "Verify",
};

const STAGE_ORDER = ["input_guardrails", "retrieve", "retrieval_gate", "generate", "grounding"];

export function LatencyHud({ timings }: { timings: StageTimings | null }) {
  const total = timings ? Object.values(timings.stages).reduce((a, b) => a + b, 0) : null;
  const overBudget = total !== null && total > LATENCY_BUDGET_MS;
  const stages = timings
    ? STAGE_ORDER.filter((name) => name in timings.stages).map((name) => ({
        name,
        label: STAGE_LABELS[name] ?? name,
        ms: timings.stages[name],
      }))
    : [];
  const maxStageMs = stages.length ? Math.max(...stages.map((s) => s.ms), 1) : 1;

  return (
    <div
      className="rounded-(--r-lg) p-6"
      style={{ border: "1.5px solid var(--ink)", background: "var(--paper-100)", boxShadow: "var(--shadow-sm)" }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <span
          className="uppercase"
          style={{
            fontFamily: "var(--font-heavy)",
            fontWeight: 800,
            fontSize: "var(--step-caption)",
            letterSpacing: ".14em",
            color: "var(--rust)",
          }}
        >
          Measured window
        </span>
        <span style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}>
          budget {LATENCY_BUDGET_MS}ms
        </span>
      </div>

      <div className="mt-1 flex items-baseline gap-3">
        <span
          className="leading-none uppercase"
          style={{
            fontFamily: "var(--font-poster)",
            fontSize: "clamp(3rem, 9vw, 5.5rem)",
            color: total === null ? "var(--muted)" : "var(--ink)",
          }}
        >
          {total === null ? "—" : Math.round(total)}
        </span>
        <span style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "var(--step-lead)", color: "var(--ink-soft)" }}>
          ms
        </span>
        {overBudget && (
          <span
            className="rounded-(--r-sm) px-2 py-1 uppercase"
            style={{
              fontFamily: "var(--font-heavy)",
              fontWeight: 700,
              fontSize: "var(--step-caption)",
              letterSpacing: ".08em",
              background: "var(--rust-tint)",
              color: "var(--rust-deep)",
            }}
          >
            +{Math.round(total! - LATENCY_BUDGET_MS)}ms over
          </span>
        )}
      </div>

      <div className="mt-6 flex flex-col gap-3">
        {stages.length === 0 && (
          <div style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}>No request yet</div>
        )}
        {stages.map((stage) => (
          <div key={stage.name} className="flex items-center gap-3">
            <span className="w-32 shrink-0" style={{ fontSize: "var(--step-caption)", color: "var(--ink-soft)" }}>
              {stage.label}
            </span>
            <div className="h-3 flex-1 overflow-hidden rounded-full" style={{ background: "var(--paper-300)" }}>
              <div
                className="h-full rounded-full transition-[width] duration-300"
                style={{ width: `${Math.min(100, (stage.ms / maxStageMs) * 100)}%`, background: "var(--rust)" }}
              />
            </div>
            <span
              className="w-16 shrink-0 text-right tabular-nums"
              style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}
            >
              {stage.ms.toFixed(1)}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
