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

export default function Home() {
  const [language, setLanguage] = useState(AUTO_DETECT.sarvamCode);
  const [strategy, setStrategy] = useState<string | null>(null);
  const { status, partialTranscript, finalTranscript, response, error, start, stop } =
    useVoiceSession();

  const handleMicClick = () => {
    void start(language, strategy);
  };

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-6 py-10">
      <header className="flex flex-col gap-1">
        <h1 className="font-mono text-sm uppercase tracking-widest text-text-faint">
          RAGInGoa
        </h1>
        <p className="text-2xl font-semibold text-text">Voice RAG, measured.</p>
      </header>

      <div className="flex flex-wrap gap-4">
        <LanguageSelector value={language} onChange={setLanguage} />
        <StrategySelector value={strategy} onChange={setStrategy} />
      </div>

      <section className="flex flex-col items-center gap-4 rounded-lg border border-border bg-bg-panel px-6 py-10">
        <MicButton status={status} onStart={handleMicClick} onStop={stop} />
        <span className="font-mono text-xs uppercase tracking-wider text-text-dim">
          {STATUS_LABEL[status]}
        </span>

        <div className="min-h-[3rem] w-full max-w-xl text-center">
          {partialTranscript && (
            <p className="text-lg text-text-dim">{partialTranscript}</p>
          )}
          {!partialTranscript && finalTranscript && (
            <p className="text-lg text-text">{finalTranscript}</p>
          )}
        </div>

        {error && (
          <p className="rounded border border-danger/40 bg-danger/10 px-3 py-2 font-mono text-xs text-danger">
            {error}
          </p>
        )}
      </section>

      <LatencyHud timings={response?.timings ?? null} />

      {response &&
        (response.outcome === "answered" ? (
          <AnswerPanel response={response} />
        ) : (
          <RefusalPanel response={response} />
        ))}
    </main>
  );
}
