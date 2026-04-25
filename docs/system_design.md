# System Design Document: Gabeo RCM Pipeline

## 1. Problem Understanding

The US healthcare RCM (Revenue Cycle Management) domain is plagued by 
"Information Asymmetry." Payers deny claims using standardized CARC/RARC 
codes, but the true reason for denial is often buried in the original claim 
data (837) vs. the adjudication response (835).

A billing team member today must manually cross-reference both documents, 
look up the CARC code meaning, assess whether the denial is valid, decide 
whether to appeal, and write the appeal letter — all for a single claim. 
At scale, this process is slow, expensive, and inconsistent.

Our system solves this by:

- **CARC-Aware Parsing**: Translating cryptic adjustment codes into 
  human-readable billing logic with full 835/837 field context.
- **Rule-Logic Fusion**: Recognizing that some denials are purely 
  data-driven (timely filing date math, duplicate detection) while others 
  require clinical reasoning (medical necessity, modifier disputes).
- **Hybrid Verdict Generation**: Combining deterministic Python rules with 
  LLM reasoning — each handling the domain where it is most reliable.
- **Batch Intelligence**: Grouping denials into actionable clusters so 
  billing teams can prioritize recovery efforts by dollar value rather than 
  processing claims one at a time.

---

## 2. Technical Architecture

### 5-Layer Hybrid Model

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — Ingestion                                    │
│  Joins 835 + 837 on ClaimID · validates fields          │
│  Enriches with CARC lookup dictionary                   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  LAYER 2 — Rule Engine (Deterministic Pre-Analysis)     │
│  Date arithmetic · remark code classification           │
│  Modifier checks · claim frequency interpretation       │
│  Outputs structured JSON flags injected into LLM prompt │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  LAYER 3 — Pattern Matching (TurboStore)                │
│  4-bit scalar quantized vector search (CPU-pinned)      │
│  Top-5 similar historical claims retrieved              │
│  Few-shot context injected into LLM prompt              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  LAYER 4 — LLM Reasoning (DeepSeek-R1:8B via Ollama)   │
│  Synthesizes rule flags + historical context            │
│  Outputs: root_cause · recoverability · confidence      │
│           evidence · recommended_action                 │
│  R1 Consistency Guard enforces deterministic overrides  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  LAYER 5 — Batch Intelligence & Output                  │
│  KMeans clustering on denied claim feature vectors      │
│  LLM-generated recovery opportunity statements          │
│  Systemic trend reporting (Payer + Procedure + CARC)    │
│  Appeal letter generation per claim                     │
│  FastAPI dashboard · CLI · JSON reports                 │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Normalization**: The `Loader` class performs a primary key join on 
   `pc_ClaimID` (835) and `ec_ClaimNo` (837), producing a unified 
   `ClaimRecord` dataclass with all fields from both documents. The 
   `CarcLookup` module enriches each record with human-readable CARC 
   descriptions loaded from `data/raw/carc_reference.txt`.

2. **Deterministic Gating**: The `RootCauseAnalyzer` runs a pre-analysis 
   pass before any LLM call. Hard rules fire first:
   - CARC 29: computes exact days elapsed, detects secondary claims via 
     `ec_PatientRelationship`, checks `ec_DelayReasonCode` presence
   - CARC 18: evaluates `ec_ClaimFrequency` (7/8 = recoverable, 1 + 
     `pc_OrigRefNo` = not_recoverable at 0.9)
   - CARC 96: matches remark codes against `NON_COVERED_CONFIRMED_REMARKS` 
     set `{N20, N130, N95}`
   - CARC 253: immediate `not_recoverable` at 1.0, bypasses LLM entirely
   - CARC 252: immediate `recoverable` at 0.9, bypasses LLM entirely
   
   Output is a structured `rule_engine_flags` JSON block injected into 
   the LLM prompt.

3. **Conflict Resolution**: The `OutputParser` applies a two-tier override 
   after parsing the LLM response:
   - Rule engine confidence ≥ 0.9 → rule engine recoverability verdict 
     overrides LLM verdict
   - `recoverability_override: true` flag → verdict locked regardless of 
     confidence
   - R1 Consistency Guard: forces `confidence = 1.0` when both rule engine 
     and LLM agree on a deterministic denial (e.g., confirmed late filing)
   - `recommended_action` is never overridden — LLM always generates the 
     professional narrative

4. **Semantic Retrieval**: `TurboStore` performs a dot-product similarity 
   search against the quantized index. Top-5 most similar historical claims 
   are retrieved and formatted as a few-shot context block injected into 
   the P1 prompt. The `PatternMatcher` builds the semantic string from 
   CARC code, payer, procedure, diagnosis, and insurance type.

5. **LLM Synthesis**: DeepSeek-R1:8B receives the claim data, rule engine 
   flags, and historical context. `<think>` tags are stripped 
   programmatically by `OutputParser` using regex before JSON extraction. 
   The model outputs a 6-key JSON object — no internal fields exposed.

---

## 3. Vector Storage & Quantization

**Embedding model**: `snowflake-arctic-embed-xs` (384 dimensions)  
**Execution**: CPU-pinned via `device="cpu"` — preserves 100% of RTX 4060 
8GB VRAM for DeepSeek-R1 inference, preventing OOM errors during concurrent 
processing.

**TurboStore quantization pipeline**:
- Float32 embeddings (384 × 4 bytes = 1,536 bytes/claim) generated by 
  arctic-xs
- Scalar 4-bit MSE quantization reduces to ~192 bytes/claim — ~87.5% 
  memory reduction vs Float32
- Quantized vectors stored in memory-mapped binary file `index.tq`
- Metadata (claim ID, recoverability, payer, CARC) stored in `meta.json`
- Search: dot-product similarity via NumPy vectorization — average query 
  latency 84ms on CPU

**Design rationale**: ChromaDB and FAISS were evaluated. ChromaDB requires 
a background server process conflicting with the offline requirement. FAISS 
does not support 4-bit scalar quantization natively. TurboStore was built 
to satisfy both constraints simultaneously.

---

## 4. AI/ML Quality

### Prompt Engineering

All prompts are stored as plain text templates in `/prompts/` with 
`{placeholder}` syntax. Three prompt files:

- `p1_root_cause.txt` — denial root cause analysis, CARC-specific 
  instruction blocks, confidence calibration table, output schema enforcement
- `p3_cluster_summary.txt` — cluster opportunity statements with 
  representative claim exemplars and CPT human-readable descriptions
- `appeal_letter.txt` — formal appeal letter generation with CARC-specific 
  counter-argument instructions and hard rules preventing clinical fact 
  invention

**Reasoning scaffolding approach**:

- **Expert Calibration**: Confidence scores are guided by a strictly 
  defined rubric — 1.0 for deterministic rule matches, 0.4–0.5 for 
  insufficient data — with CARC-specific bands for CARC 50 (0.65–0.8 
  depending on secondary diagnosis presence)
- **Conditional Logic Blocks**: Each CARC code has its own instruction 
  block in the prompt. DeepSeek-R1 reads the CARC code, finds its block, 
  and follows CARC-specific rules rather than applying general logic
- **Few-Shot Context Injection**: Similar historical claims injected as 
  structured context with explicit instruction: "Use these only to assess 
  recoverability likelihood. Do not copy their field values into this 
  claim's analysis."
- **Prohibited Behaviors Section**: Explicit list of forbidden outputs 
  (`rule_engine_flags`, `similar_claims`, `<think>` tags) at the end of 
  the prompt — R1 anchors strongly to prohibition lists

### Hallucination Prevention

**Factual Anchoring**: The LLM is provided with `rule_engine_flags` 
containing pre-computed facts. The prompt instructs: "Trust these, do not 
recalculate." For CARC 29, the LLM is explicitly prohibited from 
recalculating days elapsed — it uses the `filed_late` boolean from the 
rule engine.

**Secondary Claim Protection**: For Medicare secondary claims 
(`ec_PatientRelationship = "18"`), the prompt contains a non-negotiable 
instruction: "Do NOT calculate days elapsed from service date. The filing 
window starts from the primary payer's EOB date, which is NOT present in 
this claim data." This prevents the LLM from applying standard CARC 29 
logic to a case where it cannot have sufficient information.

**Context Safety**: The few-shot context block includes the instruction 
"Do not copy their field values into this claim's analysis" — prevents 
historical claim values from bleeding into the current claim's evidence array.

**Strict Output Schema**: The JSON schema is enforced at the prompt level 
(6 keys only, named explicitly) and at the code level (`OutputParser` 
validates structure, strips forbidden keys, falls back to regex extraction 
on parse failure).

### Model Selection Rationale

DeepSeek-R1:8B was selected over Llama-3:8B for its internal 
chain-of-thought reasoning capability. RCM denial analysis requires 
conditional reasoning — "if this diagnosis, and this payer, and no prior 
auth, then..." — which R1 handles more reliably than standard 
instruction-tuned models.

Running locally via Ollama eliminates per-claim API costs and ensures full 
PHI compliance. At cloud API rates (~$0.005/claim on GPT-4o), a 10,000 
denial/day operation would cost ~$1,500/month. The local model amortizes 
hardware cost within weeks.

**Trade-off**: 13.52s average inference latency vs ~2s for a cloud API 
call. Acceptable for batch billing workflows where throughput matters more 
than real-time response.

---

## 5. Engineering Quality & Practical Thinking

### Error Handling

- **Ollama offline**: `OllamaClient` raises a descriptive `RuntimeError` 
  with connection details — no silent failures
- **JSON parse failure**: `OutputParser` logs the raw LLM response, 
  attempts regex-based `recoverability` extraction as fallback, raises 
  `ValueError` with the raw response included in the message
- **File not found**: All file operations raise `FileNotFoundError` with 
  the full path in the message
- **VRAM contention**: `asyncio.Lock` in `OllamaClient` serializes LLM 
  calls on single-GPU setups — prevents OOM errors during batch processing
- **No bare except clauses**: All exception handling is typed and specific

### Performance Benchmarks

Hardware: Intel i7-13th Gen (16 cores) · RTX 4060 Laptop GPU (8GB VRAM) 
· 15.65GB RAM · Windows 10

| Component | Metric | Value |
|---|---|---|
| Rule engine (deterministic) | Avg latency | 0.1ms |
| TurboStore semantic search | Avg latency | 84ms |
| DeepSeek-R1 LLM inference | Avg latency | 13.52s |
| End-to-end per claim (P1+P2) | Avg latency | 14.62s |
| Full batch — 35 claims | Wall-clock time | 8.5 minutes |
| Estimated throughput | Claims/hour | ~246 |
| Infrastructure cost | Per claim | $0.00 |

### Production Readiness

**SQLite Persistence**: Every `ClaimRecord` and `DenialAnalysis` is 
persisted to `data/claims.db` after processing. The trend reporter 
performs dimensional aggregation directly against SQLite — no LLM 
involvement for pattern queries, keeping trend report generation under 
100ms regardless of dataset size.

**Modular Design**: Each pipeline stage is independently importable and 
testable. The `app/` FastAPI layer imports from `src/` but `src/` has 
zero dependency on `app/` — the pipeline runs headlessly via CLI without 
the dashboard.

**Web Dashboard**: The `app/` directory uses a clean router/config 
separation. `routers/claims.py` exposes four API endpoints (`/api/stats`, 
`/api/claims`, `/api/trends`, `/api/clusters`) consumed by a Vanilla JS 
frontend with zero npm dependencies. Clustering and trend calculations run 
live on dashboard load against the current database state.

**Test Coverage**: 11 test files covering ingestion, rule engine, output 
parser, pattern matching, clustering, trend reporting, evaluation metrics, 
benchmark math, and synthetic data integrity. All tests pass.

**Offline Compliance**: Zero external network calls during inference. 
Ollama serves the model locally. TurboStore is a local binary file. 
SQLite is embedded. No patient data leaves the machine at any point.

---

## 6. Known Limitations & Future Work

**Hardcoded filing windows**: Commercial payer filing limits (90–180 days) 
are constants in `p1_root_cause.py`. Payer-specific windows vary and 
require a dynamic policy knowledge base in production.

**Partial COB support**: Secondary claim detection routes CARC 29 cases 
to `needs_review` correctly, but full multi-payer coordination of benefits 
adjudication sequences are not modeled.

**No fine-tuning**: DeepSeek-R1:8B has no domain-specific fine-tuning on 
CARC-labeled claim data. The rule engine compensates for factual accuracy, 
but payer-specific behavioral nuances (e.g., Aetna vs UHC medical 
necessity criteria for the same CPT code) are outside the model's training 
distribution.

**TurboStore concurrent writes**: No write lock for concurrent upserts. 
Acceptable for single-user batch processing. Production requires a 
write-ahead log or migration to Qdrant/Weaviate.

**Future work**:
- Payer Policy RAG: vector store ingesting payer policy PDFs to replace 
  hardcoded filing windows
- HL7 FHIR R4 ingestion: native support for FHIR `Claim` and 
  `ClaimResponse` resources
- LoRA fine-tune of DeepSeek-R1:8B on CARC-labeled denial dataset
- Async task queue (Celery + Redis) for concurrent batch processing — 
  projected throughput increase from 246 to 1,000+ claims/hour on same 
  hardware
- TurboStore retrieval segmented by `missing_severity` for CARC 16 
  claims — eliminates batch context sensitivity for minor omission cases