"use client";

import { useState } from "react";
import { AnswerPanel } from "@/components/AnswerPanel";
import { LanguageSelector } from "@/components/LanguageSelector";
import { LatencyHud } from "@/components/LatencyHud";
import { MicButton, STATUS_LABEL } from "@/components/MicButton";
import { RefusalPanel } from "@/components/RefusalPanel";
import { StrategySelector } from "@/components/StrategySelector";
import { useVoiceSession } from "@/hooks/useVoiceSession";
import { AUTO_DETECT } from "@/lib/languages";

function SectionHead({ num, title, body }: { num: string; title: string; body?: string }) {
  return (
    <div className="mb-6">
      <span
        className="block"
        style={{ fontFamily: "var(--font-poster)", fontSize: "var(--step-caption)", color: "var(--rust)", letterSpacing: ".1em" }}
      >
        {num}
      </span>
      <h2
        className="uppercase"
        style={{ fontFamily: "var(--font-poster)", fontSize: "var(--step-h1)", letterSpacing: ".01em", lineHeight: 0.95, margin: "4px 0" }}
      >
        {title}
      </h2>
      {body && <p style={{ color: "var(--ink-soft)", fontSize: "var(--step-small)", maxWidth: "52ch" }}>{body}</p>}
    </div>
  );
}

export default function Home() {
  const [language, setLanguage] = useState(AUTO_DETECT.sarvamCode);
  const [strategy, setStrategy] = useState<string | null>(null);
  const { status, partialTranscript, finalTranscript, response, error, start, stop } =
    useVoiceSession();

  const handleMicClick = () => {
    void start(language, strategy);
  };

  return (
    <main className="relative z-10 mx-auto w-full max-w-3xl flex-1 px-6 py-8">
      {/* Topbar */}
      <header
        className="flex items-center justify-between py-5"
        style={{ borderBottom: "2px solid var(--ink)" }}
      >
        <span
          className="uppercase"
          style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "var(--step-caption)", letterSpacing: ".18em" }}
        >
          RAGInGoa · Voice RAG
        </span>
        <span style={{ fontFamily: "var(--font-poster)", fontSize: "var(--step-body)", color: "var(--muted)", letterSpacing: ".04em" }}>
          HH Goa 2026
        </span>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden py-16 text-center">
        <div className="sunburst" />
        <p
          className="relative z-10 uppercase"
          style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, letterSpacing: ".06em", fontSize: "var(--step-h2)" }}
        >
          A voice question, answered in
        </p>
        <h1
          className="relative z-10 uppercase"
          style={{
            fontFamily: "var(--font-poster)",
            lineHeight: 0.86,
            fontSize: "var(--step-poster)",
            margin: "8px 0 16px",
          }}
        >
          200<span style={{ color: "var(--rust)" }}>ms</span>
        </h1>
        <p className="relative z-10 italic" style={{ fontFamily: "var(--font-serif)", fontWeight: 500, fontSize: "var(--step-lead)", color: "var(--ink-soft)" }}>
          Grounded, cited — or an honest refusal. Nothing guessed.
        </p>
        <span
          className="relative z-10 mt-2 block"
          style={{ fontFamily: "var(--font-script)", fontSize: "clamp(1.5rem, 5vw, 2.25rem)", transform: "rotate(-3deg)" }}
        >
          speak in any of 14 languages
        </span>
      </section>

      {/* 01 Listen */}
      <section className="py-14" style={{ borderTop: "2px solid var(--ink)" }}>
        <SectionHead num="01" title="Listen" body="Push to talk, pick a language, watch the transcript arrive as you speak." />

        <div className="flex flex-wrap gap-4">
          <LanguageSelector value={language} onChange={setLanguage} />
          <StrategySelector value={strategy} onChange={setStrategy} />
        </div>

        <div
          className="mt-6 flex flex-col items-center gap-4 rounded-(--r-lg) px-6 py-10"
          style={{ border: "1.5px solid var(--ink)", background: "var(--paper-100)" }}
        >
          <MicButton status={status} onStart={handleMicClick} onStop={stop} />
          <span
            className="uppercase"
            style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "var(--step-caption)", letterSpacing: ".1em", color: "var(--muted)" }}
          >
            {STATUS_LABEL[status]}
          </span>

          <div className="min-h-12 w-full max-w-xl text-center">
            {partialTranscript && <p style={{ fontSize: "var(--step-lead)", color: "var(--muted)" }}>{partialTranscript}</p>}
            {!partialTranscript && finalTranscript && (
              <p style={{ fontSize: "var(--step-lead)", color: "var(--ink)" }}>{finalTranscript}</p>
            )}
          </div>

          {error && (
            <p
              className="rounded-(--r-sm) px-3 py-2"
              style={{ border: "1.5px solid var(--rust)", background: "var(--rust-tint)", color: "var(--rust-deep)", fontSize: "var(--step-caption)" }}
            >
              {error}
            </p>
          )}
        </div>
      </section>

      {/* 02 Latency */}
      <section className="py-14" style={{ borderTop: "2px solid var(--ink)" }}>
        <SectionHead num="02" title="Latency" body="Every stage, timed, against the 200ms line — the claim, made visible." />
        <LatencyHud timings={response?.timings ?? null} />
      </section>

      {/* 03 Answer */}
      {response && (
        <section className="py-14" style={{ borderTop: "2px solid var(--ink)" }}>
          <SectionHead num="03" title="Answer" />
          {response.outcome === "answered" ? <AnswerPanel response={response} /> : <RefusalPanel response={response} />}
        </section>
      )}

      {/* Footer */}
      <footer
        className="flex flex-wrap items-end justify-between gap-4 py-10"
        style={{ borderTop: "2px solid var(--ink)" }}
      >
        <span style={{ fontFamily: "var(--font-poster)", textTransform: "uppercase", fontSize: "var(--step-h2)" }}>
          RAGInGoa
        </span>
        <p style={{ fontSize: "var(--step-caption)", color: "var(--muted)", maxWidth: "44ch" }}>
          Voice-enabled RAG pipeline built for HH Goa 2026, Open Trial Task #2. Answers only from
          the retrieved corpus, or an explicit refusal.
        </p>
      </footer>
    </main>
  );
}
