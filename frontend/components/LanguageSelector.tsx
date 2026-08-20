"use client";

import { AUTO_DETECT, LANGUAGES } from "@/lib/languages";

interface Props {
  value: string;
  onChange: (sarvamCode: string) => void;
  detectedLanguage?: string | null;
}

export function LanguageSelector({ value, onChange, detectedLanguage }: Props) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[11px] uppercase tracking-wider text-text-faint">
        language
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-border-strong bg-bg-panel-raised px-3 py-2 font-mono text-sm text-text"
      >
        <option value={AUTO_DETECT.sarvamCode}>{AUTO_DETECT.label}</option>
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.sarvamCode}>
            {lang.label} — {lang.nativeLabel}
          </option>
        ))}
      </select>
      {value === AUTO_DETECT.sarvamCode && detectedLanguage && (
        <span className="font-mono text-[11px] text-accent">detected: {detectedLanguage}</span>
      )}
    </label>
  );
}
