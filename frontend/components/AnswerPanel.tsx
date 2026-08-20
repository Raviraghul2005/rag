"use client";

import { useState } from "react";
import type { PipelineResponse } from "@/lib/types";

export function AnswerPanel({ response }: { response: PipelineResponse }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const expandedChunk = response.cited_chunks.find((c) => c.chunk_id === expandedId) ?? null;

  return (
    <div className="rounded-lg border border-ok/30 bg-bg-panel p-5">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-wider text-ok">answered</span>
        <div className="flex items-center gap-3 font-mono text-xs text-text-faint">
          {response.provider && <span>{response.provider}</span>}
          {response.grounding_score !== null && (
            <span title="Entailment probability from grounding verification">
              grounded {(response.grounding_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      <p className="mt-3 text-lg leading-relaxed text-text">{response.answer}</p>

      {response.cited_chunks.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {response.cited_chunks.map((chunk) => (
            <button
              key={chunk.chunk_id}
              type="button"
              onClick={() =>
                setExpandedId(expandedId === chunk.chunk_id ? null : chunk.chunk_id)
              }
              aria-expanded={expandedId === chunk.chunk_id}
              className="rounded border border-border-strong bg-bg-panel-raised px-2 py-1 font-mono text-xs text-text-dim hover:border-accent hover:text-accent"
            >
              [{chunk.chunk_id}]
            </button>
          ))}
        </div>
      )}

      {expandedChunk && (
        <div className="mt-3 rounded border border-border bg-bg p-3">
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-text-faint">
            <span>lang {expandedChunk.language}</span>
            <span>dense {expandedChunk.dense_score.toFixed(3)}</span>
            <span>sparse {expandedChunk.sparse_score.toFixed(3)}</span>
            <span>fused {expandedChunk.fused_score.toFixed(3)}</span>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-text-dim">{expandedChunk.text}</p>
        </div>
      )}
    </div>
  );
}
