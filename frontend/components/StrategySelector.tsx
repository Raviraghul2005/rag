"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/config";
import { STRATEGY_LABELS } from "@/lib/types";
import type { StrategiesResponse } from "@/lib/types";

interface Props {
  value: string | null;
  onChange: (strategy: string) => void;
}

// spec §15 item 5: "cheap to build, and it proves all five strategies genuinely
// exist." Populated from GET /strategies rather than a hardcoded list of five, so the
// selector only ever offers a strategy the backend actually has an index loaded for.
export function StrategySelector({ value, onChange }: Props) {
  const [available, setAvailable] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(apiUrl("/strategies"))
      .then((res) => res.json())
      .then((data: StrategiesResponse) => {
        if (cancelled) return;
        setAvailable(data.available);
        if (!value && data.available.length > 0) {
          onChange(data.default && data.available.includes(data.default) ? data.default : data.available[0]);
        }
      })
      .catch(() => {
        /* backend not reachable yet — selector just stays empty */
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[11px] uppercase tracking-wider text-text-faint">
        chunking strategy
      </span>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={available.length === 0}
        className="rounded border border-border-strong bg-bg-panel-raised px-3 py-2 font-mono text-sm text-text disabled:opacity-50"
      >
        {available.length === 0 && <option value="">loading…</option>}
        {available.map((name) => (
          <option key={name} value={name}>
            {STRATEGY_LABELS[name] ?? name}
          </option>
        ))}
      </select>
    </label>
  );
}
