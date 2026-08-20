"use client";

import { useState } from "react";
import type { PipelineResponse } from "@/lib/types";

export function AnswerPanel({ response }: { response: PipelineResponse }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const expandedChunk = response.cited_chunks.find((c) => c.chunk_id === expandedId) ?? null;

  return (
    <div
      className="rounded-(--r-lg) p-6"
      style={{ border: "2px solid var(--ink)", background: "var(--paper-100)" }}
    >
      <div className="flex items-center justify-between">
        <span
          className="uppercase"
          style={{
            fontFamily: "var(--font-heavy)",
            fontWeight: 800,
            fontSize: "var(--step-caption)",
            letterSpacing: ".14em",
          }}
        >
          Answered
        </span>
        <div className="flex items-center gap-3" style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}>
          {response.provider && <span>{response.provider}</span>}
          {response.grounding_score !== null && (
            <span title="Entailment probability from grounding verification">
              grounded {(response.grounding_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      <p className="mt-4" style={{ fontSize: "var(--step-lead)", lineHeight: 1.5, color: "var(--ink)" }}>
        {response.answer}
      </p>

      {response.cited_chunks.length > 0 && (
        <div className="mt-5 flex flex-wrap gap-2">
          {response.cited_chunks.map((chunk) => (
            <button
              key={chunk.chunk_id}
              type="button"
              onClick={() => setExpandedId(expandedId === chunk.chunk_id ? null : chunk.chunk_id)}
              aria-expanded={expandedId === chunk.chunk_id}
              className="rounded-(--r-sm) px-3 py-1 uppercase"
              style={{
                fontFamily: "var(--font-heavy)",
                fontWeight: 700,
                fontSize: "var(--step-caption)",
                letterSpacing: ".06em",
                background: expandedId === chunk.chunk_id ? "var(--rust)" : "var(--rust-tint)",
                color: expandedId === chunk.chunk_id ? "var(--paper-100)" : "var(--rust-deep)",
              }}
            >
              [{chunk.chunk_id}]
            </button>
          ))}
        </div>
      )}

      {expandedChunk && (
        <div className="mt-4 rounded-(--r-md) p-4" style={{ border: "1.5px solid var(--ink)", background: "var(--paper-200)" }}>
          <div
            className="flex flex-wrap gap-x-4 gap-y-1"
            style={{ fontSize: "var(--step-caption)", color: "var(--muted)" }}
          >
            <span>lang {expandedChunk.language}</span>
            <span>dense {expandedChunk.dense_score.toFixed(3)}</span>
            <span>sparse {expandedChunk.sparse_score.toFixed(3)}</span>
            <span>fused {expandedChunk.fused_score.toFixed(3)}</span>
          </div>
          <p className="mt-2 italic" style={{ fontFamily: "var(--font-serif)", fontSize: "var(--step-body)", color: "var(--ink-soft)" }}>
            {expandedChunk.text}
          </p>
        </div>
      )}
    </div>
  );
}
