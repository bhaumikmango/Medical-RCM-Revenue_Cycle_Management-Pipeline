# System Design Document: Gabeo RCM Pipeline

## 1. Problem Understanding
The US healthcare RCM (Revenue Cycle Management) domain is plagued by "Information Asymmetry." Payers deny claims using standardized CARC/RARC codes, but the *true* reason for denial is often buried in the original claim data (837) vs. the adjudication response (835). 

Our system solves this by:
- **CARC-Aware Parsing**: Translating cryptic codes into human-readable billing logic.
- **Rule-Logic Fusion**: Recognizing that while some denials are purely data-driven (Timely Filing), others are behavioral (Medical Necessity).

## 2. Technical Architecture

### Data Flow
1. **Normalization**: The `Loader` class performs a "Primary Key Join" on `pc_ClaimID` (835) and `ec_ClaimNo` (837).
2. **Deterministic Gating**: The `RootCauseAnalyzer` runs a pre-check. If a claim violates a hard rule (e.g., CARC 253 - Sequestration), it is flagged with 1.0 confidence immediately.
3. **Semantic Retrieval**: The `TurboStore` performs a similarity search using a 4-bit MSE quantized index. This retrieves the "Top 3" most similar historical outcomes.
4. **LLM Synthesis**: DeepSeek-R1 receives the current claim, the deterministic flags, and the historical context. It reasons through the conflict and outputs a JSON verdict.

### Vector Storage & Quantization
To maintain high performance on standard hardware:
- **Embeddings**: `snowflake-arctic-embed-xs` (384 dimensions).
- **Quantization**: Scalar 4-bit quantization reduces memory footprint by ~75% compared to Float32.
- **Search**: Dot-product similarity optimized via NumPy vectorization.

## 3. AI/ML Quality

### Prompt Engineering
Our prompts utilize **Reasoning Scaffolding**:
- **Expert Calibration**: Confidence scores are guided by a strictly defined rubric (1.0 for facts, 0.4-0.5 for data gaps).
- **Context Injection**: Similar claims are provided as "Few-Shot" examples, helping the model recognize payer-specific patterns.

### Hallucination Prevention
- **Factual Anchoring**: The LLM is provided with `rule_engine_flags`. If the rule engine calculates a late filing, the LLM is instructed to treat that as an absolute truth.
- **Strict Output Schema**: Programmatic validation of the JSON response ensures downstream systems can always parse the recoverability verdict.

## 4. Engineering Quality & Practical Thinking

### Error Handling
- **Fallback Parsing**: If the LLM produces invalid JSON, the `OutputParser` uses a regex-based fallback to extract the `recoverability` status.
- **Lock Contention**: `asyncio.Lock` is used during LLM generation to prevent VRAM contention on local systems.

### Production Readiness
- **SQLite Persistence**: Ensures that every analysis is durable and can be audited later.
- **Modular Design**: Each stage of the pipeline (Ingest -> Analysis -> Storage) is decoupled.
- **Web Dashboard Architecture**: The `app/` directory uses a clean router/config pattern separating the API logic from the core processing logic in `src/`. It uses FastAPI for high-performance asynchrony and Vanilla JS/CSS for a lightweight, zero-dependency frontend.
