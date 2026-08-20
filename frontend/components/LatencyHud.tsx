import { LATENCY_BUDGET_MS } from "@/lib/config";
import type { StageTimings } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  input_guardrails: "guardrails",
  retrieve: "retrieve",
  retrieval_gate: "gate",
  generate: "generate",
  grounding: "verify",
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
    <div className="rounded-lg border border-border bg-bg-panel p-5">
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
          measured window
        </span>
        <span className="font-mono text-xs text-text-faint">
          budget {LATENCY_BUDGET_MS}ms
        </span>
      </div>

      <div className="mt-2 flex items-baseline gap-3">
        <span
          className={`font-mono text-6xl font-bold tabular-nums leading-none ${
            total === null ? "text-text-faint" : overBudget ? "text-danger" : "text-ok"
          }`}
        >
          {total === null ? "—" : Math.round(total)}
        </span>
        <span className="font-mono text-xl text-text-dim">ms</span>
        {overBudget && (
          <span className="font-mono text-xs text-danger">
            +{Math.round(total! - LATENCY_BUDGET_MS)}ms over
          </span>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-2">
        {stages.length === 0 && (
          <div className="font-mono text-xs text-text-faint">no request yet</div>
        )}
        {stages.map((stage) => (
          <div key={stage.name} className="flex items-center gap-3">
            <span className="w-20 shrink-0 font-mono text-xs text-text-dim">{stage.label}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-bg-panel-raised">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300"
                style={{ width: `${Math.min(100, (stage.ms / maxStageMs) * 100)}%` }}
              />
            </div>
            <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums text-text-dim">
              {stage.ms.toFixed(1)}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
