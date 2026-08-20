// Mirrors backend/app/models/pipeline.py and backend/app/models/retrieval.py.
// Kept hand-in-sync deliberately rather than codegenned — the backend response shape
// is small and stable enough that a generation step would cost more than it saves.

export type PipelineOutcome = "answered" | "refused" | "unavailable";

export interface StageTimings {
  stages: Record<string, number>;
}

export interface ScoredChunk {
  chunk_id: string;
  text: string;
  dense_score: number;
  sparse_score: number;
  fused_score: number;
  colbert_score: number | null;
  language: string;
  doc_id: string;
}

export interface PipelineResponse {
  request_id: string;
  answer: string | null;
  cited_chunk_ids: string[];
  cited_chunks: ScoredChunk[];
  sufficient: boolean;
  timings: StageTimings;
  outcome: PipelineOutcome;
  refusal_reason: string | null;
  retrieval_top_score: number | null;
  retrieval_margin: number | null;
  grounding_score: number | null;
  provider: string | null;
  reasoning_tokens: number | null;
  strategy_used: string | null;
}

export interface StrategiesResponse {
  available: string[];
  default: string;
}

// WebSocket messages received from /ws/transcribe — see backend/app/stt/voice_session.py.
export type ServerMessage =
  | { type: "partial_transcript"; text: string }
  | { type: "final_transcript"; text: string }
  | { type: "answer"; payload: PipelineResponse }
  | { type: "stt_error"; message: string };

export const STRATEGY_LABELS: Record<string, string> = {
  fixed_256_overlap_64: "Fixed window",
  recursive_512: "Recursive",
  semantic_breakpoint: "Semantic breakpoint",
  late_chunking: "Late chunking",
  metadata_aware: "Metadata-aware",
};

export const REFUSAL_LABELS: Record<string, string> = {
  empty_transcript: "No speech detected",
  too_short: "Too short to answer",
  unsafe_content: "Unsafe content blocked",
  prompt_injection: "Instruction-override attempt blocked",
  unrecognized_language: "Language not recognized",
  no_results: "Nothing relevant found in the corpus",
  low_absolute_confidence: "No confident match in the corpus",
  low_margin_confidence: "Too many similar, low-confidence matches — likely off-topic",
  insufficient_context: "Retrieved passages don't contain the answer",
  ungrounded: "Generated answer wasn't supported by the retrieved passages",
  generation_unavailable: "Answer generation is temporarily unavailable",
  no_answer_to_verify: "No answer was produced to verify",
  no_context_to_verify_against: "No context was retrieved to verify against",
};
