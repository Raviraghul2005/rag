# RAGInGoa — Requirements Specification

**Project:** Voice-Enabled RAG Pipeline
**Submission:** HH Goa 2026, Open Trial Task #2
**Builder:** Ravi Raghul (solo)
**Deadline:** 22 August 2026, 11:59 PM IST — **no resubmissions**

---

## 0. How to use this document

You are building this system with Claude Code. Read this document fully before writing any code.

Rules of engagement:

1. **Do not invent API signatures.** For Sarvam, Groq, Cerebras, FAISS and BGE-M3, fetch the current official docs and verify endpoint names, parameter names and response shapes before writing integration code. This document specifies *behaviour and contracts*, not vendor call syntax.
2. **Build in the phase order in §19.** Each phase has a definition of done. Do not start a phase before the previous one passes its check.
3. **The two tables in §17 are the submission.** The running app exists to prove those tables are real. When you have to choose between a feature and the quality of those tables, choose the tables.
4. **Every claim in the README must be reproducible** by running a command in the repo. If it cannot be reproduced, delete the claim.
5. When something in this spec turns out to be wrong or impossible on the target hardware, **say so and propose the alternative** — do not silently substitute.

---

## 1. What is being built

A system where a user speaks a question in an Indian language, and receives a grounded, cited answer generated only from a retrieved corpus — or an explicit refusal when the corpus does not support an answer.

```
Microphone
   ↓ (streaming audio)
Sarvam Saaras v3 STT  ─── partial transcripts ──┐
   ↓ (final transcript)                          │
   ↓                              speculative retrieval starts early
┌──────────── MEASURED LATENCY WINDOW ────────────┐
│  Input guardrails                               │
│  BGE-M3 encode (dense + sparse + colbert)       │
│  Hybrid retrieval (FAISS dense + sparse)        │
│  ColBERT late-interaction rerank                │
│  Abstention gate                                │
│  LLM generation (Groq / Cerebras)               │
│  Grounding verification                         │
└─────────────────────────────────────────────────┘
   ↓
Answer + citations + per-stage latency breakdown
```

---

## 2. Non-negotiable requirements (from the official task PDF)

These six are the scoring rubric. Every one must be visibly, verifiably satisfied.

| # | Official requirement | How this build satisfies it |
|---|---|---|
| 1 | STT must be Sarvam **or** ElevenLabs | Sarvam Saaras v3, streaming |
| 2 | Chunking must be "vast" — not one naive fixed-size split | 5 strategies, benchmarked head-to-head (§6) |
| 3 | Chunking + retrieval + everything through to final output **under 200ms** | Measured window defined in §13, per-stage budget in §13.2 |
| 4 | Submit **P50 / P70 / P100** across a reasonable number of queries | ≥500-query replay harness, reproducible (§13, §14) |
| 5 | Run inside a **proper harness** — tool calls, retries, structured I/O, error recovery | §12 |
| 6 | **Guardrails** — off-topic, unsafe, hallucination, ungrounded answers; know when *not* to answer | §11, with an abstention precision/recall curve |

Plus deliverables: GitHub repo, live working link, 90-second process video, demo video, both videos posted to Instagram **and** X, at least one public Instagram account, `#RAGInGoa` on every post, submission form filled.

---

## 3. Decisions to confirm before Phase 1

These change the build. Confirm each with the user before proceeding.

| Decision | Default assumed here | Impact if changed |
|---|---|---|
| **GPU Space (T4, ~$0.40/hr) vs free CPU** | **GPU assumed** | On CPU, BGE-M3 costs 150–200ms and blows the budget → fall back to `multilingual-e5-small` ONNX int8, drop ColBERT rerank, benchmark BGE-M3 offline only |
| Cerebras key available | Assumed yes | Without it, the 500-query benchmark is rate-limited by Groq's ~6k TPM; add backoff and expect long runs |
| Language scope | All 14 indexed, ~10 demoed live | Reducing to 6 cuts indexing time substantially |
| Contextual-retrieval chunking (LLM call per chunk) | Included as strategy 6, optional | Best quality, most expensive indexing step; cut first if time-pressed |

---

## 4. Technology stack

| Layer | Choice | Reason |
|---|---|---|
| STT | Sarvam Saaras v3 (WebSocket streaming) | Indic-native, code-mix, partial transcripts enable speculative retrieval |
| Embeddings | BGE-M3 | One forward pass yields dense + sparse + ColBERT vectors — hybrid retrieval *and* reranking without a second model |
| Vector index | FAISS HNSW, in-process | No network hop; sub-10ms at this scale |
| Sparse index | BGE-M3 lexical weights, inverted index | Exact term / named-entity matching, critical for Indic morphology |
| Generation | Groq `llama-3.1-8b-instant` primary, Cerebras failover | Lowest TTFT; failover is also the harness story |
| Grounding check | mDeBERTa-v3 XNLI (multilingual entailment) | ~15ms on GPU, multilingual, far cheaper than LLM-as-judge |
| API | FastAPI + Pydantic | Structured I/O requirement |
| Frontend | **Next.js (App Router) + TypeScript + Tailwind** | Required by the builder; SSR/static on the edge, strong DX with Claude Code |
| Frontend hosting | **Vercel (Hobby, free)** | Global CDN, instant deploys, custom domain |
| Backend hosting | Hugging Face Space (Docker) | Persistent warm process, models stay resident in RAM, sits next to the dataset |

**Split-deployment rules (important):**

- The **browser talks directly to the FastAPI Space**, not through Next.js API routes. Reasons: Vercel Hobby kills functions at 10s, serverless routes cannot hold a WebSocket, and an extra hop adds pure latency for nothing.
- Next.js is therefore a **pure client** — App Router with client components for the mic and HUD. No server actions in the request path.
- **CORS** on FastAPI: allow only the Vercel production domain and `localhost` for dev. Not `*`.
- **`SARVAM_API_KEY` must never reach the browser.** `NEXT_PUBLIC_*` variables are compiled into the client bundle and the repo is public. The browser opens a WebSocket to **your FastAPI backend**, which proxies to Sarvam server-side with the key held in Space Secrets. Build the proxy; do not shortcut this.
- The only public env var the frontend needs is `NEXT_PUBLIC_API_BASE_URL` (the Space URL).

**Explicitly rejected** (state these in the README with reasons — the reasoning scores):
hosted embedding APIs (network hop inside the budget), hosted vector DBs (same), cross-encoder rerankers (100ms+), LLM-as-judge guardrails (seconds), TTS output (not required).

---

## 5. Data pipeline

**Source:** `ai4bharat/MSMARCO-XI` on Hugging Face. 14 language configs (as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur). ~11.45M rows, ~55.6GB total.

**Do not download the full dataset.** Stream it.

Requirements:

1. Use `datasets.load_dataset(..., streaming=True)` per language config.
2. Sample **N passages per language** (default N=50,000; configurable). Sample deterministically with a fixed seed so the corpus is reproducible.
3. Extract per row: `query`, `query_id`, `query_type`, `Answer`, and from `passages`: `Translated_passages`, `English_passages`, `is_selected`.
4. Build two artifacts:
   - **Corpus:** deduplicated passages with metadata `{passage_id, language, query_type, is_selected, source_query_id}`
   - **Eval set:** held-out `(query, relevant_passage_ids)` pairs derived from `is_selected == 1`. Minimum 500 queries, stratified across languages and `query_type`.
5. Persist corpus + eval set + built indexes to a **Hugging Face dataset repo**, not to the Space container. Space storage is wiped on rebuild; the app must pull prebuilt artifacts at startup.
6. Record and print corpus statistics (rows per language, passage length distribution, dedup rate) — these go in the README.

**Quality gate:** verify that translated passages are non-empty and script-correct per language. Log and exclude rows that fail. Report the exclusion rate.

---

## 6. Chunking module

The single highest-scoring component. Build it as a clean plugin interface so strategies are swappable and benchmarkable.

```python
class ChunkStrategy(Protocol):
    name: str
    def chunk(self, doc: Document) -> list[Chunk]: ...

@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    language: str
    query_type: str | None
    char_start: int
    char_end: int
    strategy: str
    extra: dict          # strategy-specific (e.g. parent_id, context_prefix)
```

Implement **five** strategies (six with the optional):

| # | Strategy | Spec |
|---|---|---|
| 1 | `fixed_256_overlap_64` | Token-based fixed windows, 64-token overlap. Baseline. |
| 2 | `recursive_512` | Recursive separator splitting (paragraph → sentence → word), target 512 tokens, 50-token overlap. Separators must include Devanagari danda (`।`) and other Indic sentence terminators — **not just `.`**. This detail matters and most submissions will miss it. |
| 3 | `semantic_breakpoint` | Embed sentences, split at percentile-threshold cosine-distance breakpoints. Configurable percentile (default 95). |
| 4 | `late_chunking` | Encode the **whole document** with BGE-M3 (8k context), then pool token embeddings into chunk-level vectors. Each chunk vector carries whole-document context. Chunk boundaries follow strategy 2. |
| 5 | `metadata_aware` | Chunk boundaries respect `query_type` and passage boundaries; emits filterable payload for pre-filtered ANN search. |
| 6 | `contextual_retrieval` *(optional)* | LLM writes a 1–2 sentence situating context per chunk, prepended before embedding. Expensive at index time, best quality. |

**Deliverable:** every strategy indexed and evaluated independently on the same eval set. Comparison table (§14.1) is mandatory.

---

## 7. Indexing

For each chunking strategy, build:

- **Dense index:** FAISS `HNSW32` (`efConstruction=200`, `efSearch` tunable at query time), inner-product on L2-normalized BGE-M3 dense vectors.
- **Sparse index:** inverted index over BGE-M3 lexical weights, with per-token weight lookup.
- **ColBERT store:** per-chunk multi-vector arrays, memory-mapped, loaded only for the top-K candidates at rerank time. **Do not load all ColBERT vectors into RAM.**
- **Metadata store:** SQLite or Parquet, keyed by `chunk_id`, holding text + payload for filtered search.

Requirements:

- Index build must be a standalone, resumable CLI command with progress output.
- Record and report index build time, index size on disk, and RAM footprint per strategy.
- Serialize everything to the HF dataset repo, versioned by strategy name.

---

## 8. Retrieval

```
query
 ├── BGE-M3 single encode → {dense, sparse, colbert}
 ├── dense search   (FAISS HNSW, top-50)
 ├── sparse search  (lexical weights, top-50)
 ├── fusion         (Reciprocal Rank Fusion, k=60) → top-20
 ├── ColBERT late-interaction rerank on top-20 → top-5
 └── return top-5 with scores
```

Requirements:

- **One encode call.** Never encode the query more than once per request.
- RRF constant, top-K values and `efSearch` must be config, not literals.
- Optional metadata pre-filtering by language when the STT reports a confident language ID.
- Every retrieval result carries `{chunk_id, text, dense_score, sparse_score, fused_score, colbert_score, language, doc_id}`.
- Retrieval must be independently benchmarkable without the LLM (needed for §14.1).

---

## 9. Speech-to-text and speculative retrieval

**This is the wow moment. Build it properly.**

- Sarvam Saaras v3 over WebSocket streaming, emitting partial transcripts.
- On each partial transcript exceeding a minimum token count (default 3 tokens), **fire encode + retrieval speculatively** in the background.
- Cache the speculative result keyed by transcript prefix hash.
- On final transcript: if it matches the last speculative transcript, reuse the cached retrieval — retrieval latency is then effectively zero. If it differs, discard and re-run.
- Report **speculation hit rate** in the eval package. It is a real, novel, measurable number.
- Speculative work must never block or corrupt the final answer path. If speculation errors, the final path runs normally and logs the miss.

STT latency is reported separately and is **outside** the measured 200ms window (§13.1). Do not hide it — report it in its own row.

**Language coverage:** the dataset has 14 languages; Sarvam STT does not cover all of them. State the gap explicitly in the README with a coverage table. Naming your own limitation is worth more than quietly omitting it.

---

## 10. Generation

- Primary: Groq `llama-3.1-8b-instant`. Failover: Cerebras.
- **Cap output at 60 tokens** (`max_tokens=60`). Output length is the largest single lever on total latency.
- Stream the response; the measured window ends at **final token** (§13.1) but time-to-first-token is reported as a separate row.
- Prompt requirements:
  - Answer **only** from the provided context.
  - Answer in the **same language as the question**.
  - Emit a citation to the `chunk_id`s used.
  - If context is insufficient, return the refusal token — do not guess.
  - Structured output: `{answer, cited_chunk_ids, sufficient: bool}`.
- Retrieved context must be **trimmed to a token budget** (default 1,200 tokens) before the call. This controls both latency and the Groq TPM rate limit.

---

## 11. Guardrails

Three layers. All must be individually toggleable so ablations can be run.

### 11.1 Input guardrails (~1ms, pre-retrieval)
- Language identification with confidence.
- Prompt-injection pattern detection (instruction-override phrasing, role-play framing, system-prompt extraction attempts) — applied to the **transcript**, since voice input is still untrusted input.
- Unsafe/abusive content screening.
- Empty / too-short / non-speech transcript handling.
- Outcome: `{allowed: bool, reason: str}`.

### 11.2 Retrieval-confidence gate (0ms — reuses existing scores)
Abstain when **either**:
- top-1 fused score < `TAU_ABS` (absolute threshold), **or**
- (top-1 − mean(top-2..top-5)) < `TAU_MARGIN` (margin threshold).

The margin test is the important one: a flat threshold is a guess, but a *low margin* means "many mediocre matches," which is exactly the signature of an off-topic query. Costs nothing extra.

Both thresholds must be **calibrated on labelled data**, not hand-picked. Produce the calibration curve.

### 11.3 Grounding verification (post-generation)
- mDeBERTa-v3 XNLI entailment: does the retrieved context entail the generated answer?
- Below threshold → suppress the answer, return the refusal.
- Report latency for this stage separately, and report the pipeline **both with and without** it so the reader can see the cost of safety.

### 11.4 Abstention evaluation (mandatory)
Build a labelled set of **answerable** and **unanswerable** queries:
- Answerable: eval queries with known relevant passages in the corpus.
- Unanswerable: (a) off-topic queries, (b) queries whose relevant passages were deliberately excluded from the index, (c) unsafe/injection queries.

Report **abstention precision, recall, F1, and a threshold-sweep curve.** This turns "we added guardrails" into "our system knows when not to answer, and here is the number." Almost nobody will do this.

---

## 12. Harness

Requirement #5. Build it early — it also generates the latency data for free.

Required:

1. **Typed I/O end to end.** Pydantic models at every stage boundary. No untyped dicts crossing module lines.
2. **Per-stage instrumentation.** A context manager that records monotonic-clock durations for every stage into a `StageTimings` object attached to every request. This is the source of the latency tables — never a separate ad-hoc script.
3. **Retries with exponential backoff + jitter** on all external calls (STT, Groq, Cerebras). Bounded attempts, bounded total time.
4. **Circuit breaker** on the generation provider: N consecutive failures → open circuit → route to failover → half-open probe after cooldown.
5. **Timeouts at every stage**, with a total request deadline. Exceeding a stage budget degrades gracefully rather than hanging.
6. **Graceful degradation ladder**, explicit and logged:
   - ColBERT rerank fails → serve fused results
   - Sparse index fails → dense-only
   - Grounding model fails → serve answer flagged `unverified`
   - Both LLM providers fail → return retrieved passages with an honest "generation unavailable" message
7. **Structured JSON logging** of every request: request id, transcript, language, strategy, stage timings, scores, abstention decision and reason, provider used, retries, final outcome.
8. **Deterministic replay runner:** a CLI that takes a fixed query set and regenerates the entire latency table. This is the real answer to *"not a lucky run"* — a judge can rerun your numbers.

Tool-calling: expose retrieval as a callable tool with a typed schema rather than string-concatenating context into the prompt. The task explicitly names tool calls.

---

## 13. Latency measurement

### 13.1 Window definition (state this verbatim in the README)

The official wording is: *"the full process — chunking + vector DB retrieval + everything through to final output — should complete in under 200ms."*

**Measured window: from receipt of the final transcript text → final generated token.** It includes input guardrails, query encoding, dense + sparse retrieval, fusion, ColBERT rerank, abstention gate, LLM generation, and grounding verification.

Reported **separately, outside the window**: STT latency, network round-trip to the client, audio capture, index build time.

Do not fudge this. Define it once, state it plainly, report every stage, and let the numbers stand. A clearly-defined 210ms with a full breakdown is worth far more than an undefined 190ms.

### 13.2 Target budget (GPU configuration)

| Stage | Target |
|---|---|
| Input guardrails | 1ms |
| BGE-M3 encode | 15ms |
| Dense + sparse retrieval | 10ms |
| Fusion | 1ms |
| ColBERT rerank (top-20) | 10ms |
| Abstention gate | 0ms |
| LLM generation (60 tokens) | 140ms |
| Grounding verification | 15ms |
| **Total** | **~192ms** |

Generation dominates. If the budget is missed, the levers in order are: output token cap, context token budget, provider choice, `efSearch`, rerank candidate count.

### 13.3 Methodology (must be documented)

- ≥500 queries, stratified by language and query type.
- **Cold and warm runs reported separately**, never mixed into one percentile.
- P50 / P70 / P100 reported **per stage and end to end**.
- Sample size, hardware, date, model versions, and provider stated alongside every table.
- P100 = worst single observation. Report it honestly and explain what caused the outlier (usually a provider hiccup); do not quietly trim.
- Use `time.perf_counter()`, never wall-clock.

---

## 14. Evaluation package

### 14.1 Chunking comparison (the centrepiece)

| Strategy | Recall@5 | nDCG@10 | MRR | Chunks | Index size | Build time | Query latency |
|---|---|---|---|---|---|---|---|

Per strategy, and **broken down per language**. Then a written paragraph: which strategy is served in production and *why* — including the case where the best-scoring strategy is not the one served because of latency. That reasoning is what "clear thinking" means on the selection criteria.

### 14.2 Ablations

| Configuration | Recall@5 | nDCG@10 | P50 latency |
|---|---|---|---|
| Dense only | | | |
| Dense + sparse (RRF) | | | |
| Dense + sparse + ColBERT rerank | | | |
| + metadata pre-filtering | | | |

Also ablate: embedding model (BGE-M3 vs multilingual-e5-small — quality vs latency on target hardware), and guardrails on vs off.

### 14.3 Abstention curve
Per §11.4.

### 14.4 Speculation hit rate
Percentage of queries where the speculative retrieval was reusable, and the latency saved.

All tables must be regenerable by a single documented command.

---

## 15. Frontend

**Next.js (App Router), TypeScript, Tailwind, deployed on Vercel.** One page. Its job: make the engineering legible in a 60-second demo video.

Technical requirements:

- App Router, single route. Mic capture, WebSocket handling and the latency HUD are **client components** (`"use client"`).
- Audio capture via `MediaRecorder` / `AudioWorklet`; stream chunks over a WebSocket to **your FastAPI backend**, which proxies to Sarvam. Never open a WebSocket to Sarvam from the browser.
- Partial transcripts arrive over the same socket and render as the user speaks.
- Stage timings arrive with the final response payload and drive the HUD.
- Use `NEXT_PUBLIC_API_BASE_URL` for the Space URL. No other public env vars.
- No `next/image` optimization dependency, no server actions, no middleware in the request path — the frontend must stay a thin, fast client.

Required elements:

Required elements:

1. **Push-to-talk microphone** with live waveform and streaming partial transcript appearing as the user speaks.
2. **Live per-stage latency HUD** — encode / retrieve / rerank / generate / verify, updating on every query, with the total against the 200ms line. This is what makes the claim visible rather than asserted.
3. **Answer with inline citations**, each citation expandable to the source passage and its score.
4. **Visible refusal state** — when the system abstains, show *why* (low confidence, off-topic, ungrounded, unsafe) with the actual score that triggered it. The refusal must look like a designed outcome, not an error.
5. **Strategy selector** — let the viewer switch chunking strategy live and watch retrieval change. Cheap to build, and it proves all five strategies genuinely exist.
6. **Language selector / auto-detect indicator.**

Design notes: the interface is an instrument, not a marketing page. Type and layout should read as a measurement tool — dense, precise, legible at video resolution. Latency numbers are the hero content; make them large and unmissable. Avoid the generic "AI chat app" look. Keyboard focus visible, responsive to mobile, `prefers-reduced-motion` respected.

Copy: errors state what happened and what to do. The refusal message says what the system could not verify, not "Sorry, I can't help with that."

---

## 16. Deployment

### 16.1 Frontend — Vercel

- Next.js app in `/frontend`, deployed from the same GitHub repo (set the Vercel root directory).
- Free Hobby plan. Personal/non-commercial only — fine for a hackathon submission.
- `NEXT_PUBLIC_API_BASE_URL` set in Vercel project env vars.
- **The Vercel URL is the "live working link" you submit.**
- Verify the deployed frontend reaches the Space from a clean browser with no cache, on mobile as well as desktop.

### 16.2 Backend — Hugging Face Space

- **Hugging Face Space, Docker SDK.** Container must `EXPOSE 7860`.
- All keys via Space Secrets. **Never commit a key.** The repo is public.
- Startup: pull prebuilt indexes from the HF dataset repo, load models, warm the encoder and index with a dummy query **before** marking `/health` ready.
- `/health` endpoint returning model load state, index state, and provider reachability.
- Cold start is a submission risk — a judge hitting a sleeping Space sees a 30–90s blank screen. Set up an external cron pinger against `/health`.
- Space storage is wiped on rebuild: never treat container disk as durable.

---

## 17. README (this is a scored artifact — treat it as the deliverable it is)

Required sections, in this order:

1. One-paragraph description + live link + demo video link
2. **Latency table** — headline numbers, per-stage, P50/P70/P100, with the window definition stated verbatim
3. **Chunking comparison table** + which strategy is served and why
4. Architecture diagram + request trace
5. Guardrails, with the abstention precision/recall numbers
6. Ablation tables
7. Harness design — retries, failover, degradation ladder
8. Reproduce-it-yourself commands
9. Honest limitations: STT language coverage gap, corpus subset size, what breaks at scale, what you would do with more time
10. Setup instructions

A judge should be able to score requirements 2, 3, 4 and 6 **without running anything**, in the first ninety seconds of reading. Put the tables above the fold.

---

## 18. Acceptance criteria

Ship only when every line is true:

- [ ] Voice input works end to end in ≥5 languages, via Sarvam
- [ ] All 5 (or 6) chunking strategies implemented, indexed, and evaluated
- [ ] Comparison table generated from a reproducible command
- [ ] Measured window documented and defended; per-stage P50/P70/P100 over ≥500 queries
- [ ] Cold and warm runs reported separately
- [ ] Harness: typed I/O, retries, circuit breaker, failover, degradation ladder, structured logs, replay runner
- [ ] Guardrails: all three layers, with an abstention precision/recall curve
- [ ] System demonstrably refuses an off-topic question, an unanswerable question, and an injection attempt — on camera
- [ ] Live link cold-starts clean from a fresh browser with no cache
- [ ] No secrets in the repo (scan the git history, not just the working tree)
- [ ] README complete, tables above the fold
- [ ] Both videos recorded; posts on Instagram + X with `#RAGInGoa`; ≥1 public Instagram account
- [ ] Submission form filled

---

## 19. Build order

| Phase | Work | Definition of done |
|---|---|---|
| 1 | Repo skeleton, config, typed models, stage-timer context manager, structured logging | `make test` passes; timers emit on a stub pipeline |
| 2 | Data pipeline — stream, sample, dedup, build eval set, push to HF dataset repo | Corpus stats printed; eval set has ≥500 labelled queries |
| 3 | Chunking module + all strategies | Each strategy chunks a sample doc; unit tests on boundaries incl. Indic terminators |
| 4 | Indexing — dense, sparse, ColBERT store, metadata | Indexes build and reload from the HF repo |
| 5 | Retrieval — single encode, RRF fusion, ColBERT rerank | Retrieval-only benchmark runs; §14.1 table generated |
| 6 | Generation + prompt + structured output | End-to-end text query → cited answer |
| 7 | Guardrails, all three layers + calibration | Abstention curve generated |
| 8 | Harness hardening — retries, breaker, failover, degradation | Chaos test: kill Groq → Cerebras serves; kill both → graceful |
| 9 | STT + speculative retrieval | Voice → answer works; speculation hit rate measured |
| 10 | Frontend + latency HUD | Demo-able on a phone screen |
| 11 | Deploy to Space, warm start, health check, pinger | Cold-start test from clean browser passes |
| 12 | Full benchmark run, all tables, README | Every table regenerable by one command |
| 13 | Videos, posts, form | Submitted |

**Freeze code 24 hours before the deadline.** No resubmissions means a broken live link on the 22nd is fatal in a way no amount of good engineering compensates for.

---

## 20. Do not build

Cross-encoder rerankers. Hosted vector DBs. Hosted embedding APIs. TTS output. Multi-turn conversation memory. User accounts or auth. A landing page. LLM-as-judge guardrails in the hot path. Agent frameworks. Docker Compose with five services. Any feature not traceable to a line in §2.

---

## 21. Configuration

All of the following are config, never literals in code:

```yaml
corpus:
  languages: [hi, bn, ta, te, kn, mr, gu, ml, pa, or, ur, ne, as, sa]
  passages_per_language: 50000
  seed: 42

chunking:
  active_strategy: recursive_512
  strategies: [fixed_256_overlap_64, recursive_512, semantic_breakpoint,
               late_chunking, metadata_aware]

retrieval:
  dense_top_k: 50
  sparse_top_k: 50
  rrf_k: 60
  rerank_candidates: 20
  final_top_k: 5
  ef_search: 64

guardrails:
  tau_abs: null          # calibrated, not guessed
  tau_margin: null       # calibrated, not guessed
  grounding_threshold: null
  enable_input: true
  enable_retrieval_gate: true
  enable_grounding: true

generation:
  primary: groq
  primary_model: llama-3.1-8b-instant
  failover: cerebras
  max_tokens: 60
  context_token_budget: 1200

harness:
  max_retries: 3
  stage_timeout_ms: {encode: 100, retrieve: 100, rerank: 100, generate: 2000}
  total_deadline_ms: 5000
  breaker_failure_threshold: 5
  breaker_cooldown_s: 30
```

Secrets (Space Secrets only): `SARVAM_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `HF_TOKEN`.

---

## 22. Risks

| Risk | Mitigation |
|---|---|
| Groq free tier ~6k TPM throttles the 500-query benchmark | Cerebras for benchmark runs; tight context budget; backoff; consider Developer tier (free with a card, ~10x limits) |
| BGE-M3 too slow on CPU | GPU Space, or fall back to e5-small and benchmark BGE-M3 offline only |
| Cold-start on a sleeping Space during judging | External cron pinger + warm-up before `/health` ready |
| Indexing 14 languages × 5 strategies takes longer than expected | Strategies are independent — build and evaluate incrementally, never all-or-nothing |
| Sarvam WebSocket integration friction (audio format, VAD) | Build Phase 9 against a recorded audio file first, then wire the live mic |
| Secret leaked in public repo | Pre-commit hook + history scan before submission |
| STT key exposed in the client bundle | Browser never talks to Sarvam directly — WebSocket proxied through FastAPI, key in Space Secrets only |
| Frontend deployed but backend unreachable (CORS, cold Space, wrong env var) | Test the Vercel URL from a clean browser and a phone before submitting; `/health` check wired into the UI |
| Missing the 200ms target | Levers in §13.2, in order. Failing honestly with a documented breakdown still scores; a vague claim does not |
