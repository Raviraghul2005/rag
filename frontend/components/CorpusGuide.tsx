"use client";

import { CORPUS_DESCRIPTION, examplesForLanguage, OUT_OF_SCOPE_EXAMPLES } from "@/lib/examples";
import { LANGUAGES } from "@/lib/languages";

interface Props {
  language: string;
  onPick: (question: string) => void;
  disabled?: boolean;
}

const LANGUAGE_NAME: Record<string, string> = Object.fromEntries(
  LANGUAGES.map((l) => [l.code, l.label])
);

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
  const isAuto = language === "auto";

  return (
    <div
      className="rounded-(--r-lg) p-6"
      style={{ border: "1.5px solid var(--ink)", background: "var(--paper-100)" }}
    >
      <Label>The corpus</Label>
      <p className="mt-2" style={{ fontSize: "var(--step-small)", color: "var(--ink-soft)", lineHeight: 1.6 }}>
        {CORPUS_DESCRIPTION.what}
      </p>

      <div className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
        {CORPUS_DESCRIPTION.stats.map((s) => (
          <div key={s.label}>
            <div
              style={{
                fontFamily: "var(--font-poster)",
                fontSize: "var(--step-h2)",
                lineHeight: 1,
                color: "var(--ink)",
              }}
            >
              {s.value}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--muted)" }}>{s.label}</div>
          </div>
        ))}
      </div>

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
        <Label>
          {isAuto
            ? "Try one — five languages, speak it or click to run"
            : "Try one — speak it, or click to run as text"}
        </Label>
        <div className="mt-3 flex flex-col gap-2">
          {examples.map((ex) => (
            <button
              key={ex.gloss}
              type="button"
              onClick={() => onPick(ex.text)}
              disabled={disabled}
              className="rounded-(--r-md) px-4 py-3 text-left transition-colors disabled:opacity-50"
              style={{ border: "1.5px solid var(--line-strong)", background: "var(--paper-200)" }}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span
                  style={{ fontSize: "var(--step-body)", color: "var(--ink)" }}
                  dir={ex.languageCode === "ur" ? "rtl" : "ltr"}
                >
                  {ex.text}
                </span>
                <span
                  className="shrink-0 uppercase"
                  style={{
                    fontFamily: "var(--font-heavy)",
                    fontWeight: 800,
                    fontSize: "0.62rem",
                    letterSpacing: ".1em",
                    color: "var(--rust)",
                  }}
                >
                  {LANGUAGE_NAME[ex.languageCode] ?? ex.languageCode}
                </span>
              </div>
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
