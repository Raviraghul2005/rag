"use client";

import { CORPUS_DESCRIPTION, examplesForLanguage, OUT_OF_SCOPE_EXAMPLES } from "@/lib/examples";

interface Props {
  language: string;
  onPick: (question: string) => void;
  disabled?: boolean;
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="uppercase"
      style={{
        fontFamily: "var(--font-heavy)",
        fontWeight: 800,
        fontSize: "0.7rem",
        letterSpacing: ".12em",
        color: "var(--muted)",
      }}
    >
      {children}
    </span>
  );
}

export function CorpusGuide({ language, onPick, disabled }: Props) {
  const examples = examplesForLanguage(language);

  return (
    <div
      className="rounded-(--r-lg) p-6"
      style={{ border: "1.5px solid var(--ink)", background: "var(--paper-100)" }}
    >
      <p style={{ fontSize: "var(--step-small)", color: "var(--ink-soft)", lineHeight: 1.6 }}>
        This answers from a fixed corpus:{" "}
        <strong style={{ color: "var(--ink)" }}>{CORPUS_DESCRIPTION.what}</strong>
      </p>

      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        <div>
          <Label>It answers</Label>
          <p className="mt-1.5" style={{ fontSize: "var(--step-small)", color: "var(--ink-soft)", lineHeight: 1.55 }}>
            {CORPUS_DESCRIPTION.answers}
          </p>
        </div>
        <div>
          <Label>It declines</Label>
          <p className="mt-1.5" style={{ fontSize: "var(--step-small)", color: "var(--ink-soft)", lineHeight: 1.55 }}>
            {CORPUS_DESCRIPTION.refuses}
          </p>
        </div>
      </div>

      <div className="mt-6">
        <Label>Try one — speak it, or click to run as text</Label>
        <div className="mt-3 flex flex-col gap-2">
          {examples.map((ex) => (
            <button
              key={ex.gloss}
              type="button"
              onClick={() => onPick(ex.text)}
              disabled={disabled}
              className="rounded-(--r-md) px-4 py-3 text-left transition-colors disabled:opacity-50"
              style={{
                border: "1.5px solid var(--line-strong)",
                background: "var(--paper-200)",
              }}
            >
              <span style={{ fontSize: "var(--step-body)", color: "var(--ink)" }}>{ex.text}</span>
              <span className="mt-0.5 block" style={{ fontSize: "0.7rem", color: "var(--muted)" }}>
                {ex.gloss}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6">
        <Label>These will be refused — on purpose</Label>
        <div className="mt-2 flex flex-wrap gap-2">
          {OUT_OF_SCOPE_EXAMPLES.map((q) => (
            <span
              key={q}
              className="rounded-(--r-sm) px-2.5 py-1"
              style={{
                border: "1px dashed var(--line-strong)",
                fontSize: "0.7rem",
                color: "var(--muted)",
              }}
            >
              {q}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
