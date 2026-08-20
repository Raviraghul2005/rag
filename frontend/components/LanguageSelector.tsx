"use client";

import { AUTO_DETECT, LANGUAGES } from "@/lib/languages";

interface Props {
  value: string;
  onChange: (sarvamCode: string) => void;
  detectedLanguage?: string | null;
}

export function LanguageSelector({ value, onChange, detectedLanguage }: Props) {
  return (
    <label className="flex flex-col gap-1.5">
      <span
        className="uppercase"
        style={{ fontFamily: "var(--font-heavy)", fontWeight: 800, fontSize: "0.7rem", letterSpacing: ".1em", color: "var(--muted)" }}
      >
        Language
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-(--r-md) px-3 py-2"
        style={{
          border: "1.5px solid var(--ink)",
          background: "var(--paper-100)",
          color: "var(--ink)",
          fontSize: "var(--step-small)",
        }}
      >
        <option value={AUTO_DETECT.sarvamCode}>{AUTO_DETECT.label}</option>
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.sarvamCode}>
            {lang.label} — {lang.nativeLabel}
          </option>
        ))}
      </select>
      {value === AUTO_DETECT.sarvamCode && detectedLanguage && (
        <span style={{ fontSize: "0.7rem", color: "var(--rust)" }}>detected: {detectedLanguage}</span>
      )}
    </label>
  );
}
