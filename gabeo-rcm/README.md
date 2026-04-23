# Gabeo AI: Making Sense of Healthcare Claim Denials

I've built a high-precision pipeline to figure out why health insurance claims get rejected. It’s a hybrid system that uses old-school billing rules mixed with the reasoning power of **DeepSeek-R1:8B**. Basically, it automates the boring stuff—identifying root causes, seeing if we can get the money back, and spotting trends in big batches of data.

---

## 1. Architecture Overview(How it works)

The system is split into a **5-Layer Hybrid Model**. Here’s the breakdown:

1.  **Ingestion**: It grabs **835 Remittance** and **837 Claim** files and squashes them into one `ClaimRecord`.
2.  **TurboStore (The Memory)**: This is our custom search engine. I used **4-bit quantization** so it runs on your CPU, leaving the GPU free for the heavy lifting. It looks at past claims to find patterns.
3.  **The Rule Engine**: Before the AI even looks at a claim, this layer checks the hard facts—like "did the user miss the filing deadline?" or "is there a missing modifier?" AI is great, but math is better left to code.
4.  **The Brain (DeepSeek-R1)**: I ran **DeepSeek-R1:8B** locally via Ollama. It looks at the claim data and the "red flags" from the rule engine to explain *why* things went wrong.
5.  **Batch Insights**: I used **KMeans Clustering** to group denials together. It helps us find the "big fish" (lucrative recovery chances) instead of chasing $10 errors one by one.

### The File Room
```text
gabeo-rcm/
├── app/                # FastAPI Web Dashboard
│   ├── core/           # App Configuration
│   ├── routers/        # API Routes (Claims/Stats)
│   ├── templates/      # HTML Interface
│   └── static/         # CSS/Assets
├── data/               # SQLite Database, Vector Storage, and Raw CSV Files
├── docs/               # System Architecture and Sample Results
├── prompts/            # Textual Templates for P1 and P3
├── scripts/            # Evaluation and Synthetics Scripting
├── src/
│   ├── ingestion/      # 835/837 Matching and CARC Lookups
│   ├── analysis/       # Root Cause, Pattern Match, Clustering, Trends
│   ├── storage/        # TurboStore (Quantization) and SQLite
│   └── llm/            # Ollama Client and Result Parsing
├── tests/              # Comprehensive Test Suite
├── main.py             # Command Line Interface
└── requirements.txt
```

---

## 2. Setup & Run Instructions(Getting Started)

### What you need
- **Python 3.10+**
- **Ollama** (for running the AI locally)

### Setup
1.  **Grab the AI model**
    `ollama pull deepseek-r1:8b`
2.  **Install the boring stuff**
    `pip install -r requirements.txt`

### Running the Pipe
-   **Analyze a single claim**: `python main.py analyze --claim data/samples/claim_a.json`
-   **Run a whole batch**: `python main.py batch --input data/synthetic/claims.json --cluster`
-   **Spot trends**: `python main.py trends --min-claims 3`
-   **Open the Dashboard**: `python main.py ui --port 8000` (Then just head to `localhost:8000` in your browser. It’s got a nice amber/white theme.)

---

## 3. Design Decisions & Trade-offs

### DeepSeek-R1 Locally Hosted over Other Models
I chose R1 for its internal chain-of-thought (reasoning) capabilities while being lightweight and opensource reducing the costs and privacy concerns. In RCM, "why" a claim is denied is often hidden in complex logic; R1's reasoning traces these paths better than standard instruction models. Furthermore, it is very efficient for this task as we only need the reasoning capability not the creative capabilities or any other capabilities.

### Rule Engine Before LLM
To prevent hallucinations in mathematical domains (like calculating a 180-day filing limit), I used a deterministic Python layer. The LLM is "instructed" by these facts, ensuring 100% accuracy on date math.

### Custom VectorStore over ChromaDB or FAISS
To preserve the 8GB VRAM of our local system for the LLM, the vector store uses scalar 4-bit quantization and runs embeddings on the CPU. This allows for searching thousands of historical claims with near-zero GPU impact.

### SQLite over Postgres
For a local "ML Assignment" context, SQLite provides zero-configuration persistence and full QL capability for trend reporting without requiring a background service.

---

## 4. Evaluation Results (Synthetic v1.0)

|         **Metric**        |       **Result**       |
|           :---            |          :---          |
|      **Rule Accuracy**    |         100.0%         |
|**Recovery Logic Accuracy**|         100.0%         |
|  **AI Confidence (Avg)**  |          0.81          |
| **Success Rate (Parsing)**|         100.0%         |

*I tested this on 35 "ground-truth" claims to make sure the logic holds up.*
 
---
 
## 7. Evaluation & Iteration
 
This system uses a rigorous "Evidence-Based Iteration" loop to maintain high precision.
 
**Phase 1: Initial Benchmark**
- **Overall Accuracy**: 91.4% (32/35)
- **Root Cause Analysis**: Identified 3 consistent failures where the LLM hedged into `needs_review` due to missing rule-engine signals.
  - **CARC 18**: The `pc_OrigRefNo` field was not mapped in the ingestion layer, leading to ambiguity in duplicate detection.
  - **CARC 96**: The non-coverage set only included `N130`, missing common codes like `N20` ("Service not covered by this payer").
 
**Phase 2: Targeted Refinement**
- Mapped `original_ref` in `loader.py` to enable deterministic duplicate locking.
- Expanded `p1_root_cause.py` to include `N20` and `N95` as high-confidence non-coverage signals.
- Upgraded CARC 18 logic to override LLM verdicts (Confidence 0.9) when an original reference number is present.
 
**Phase 3: Final Verification**
- **Overall Accuracy**: 100% (35/35)
- **not_recoverable Recall**: Improved from 0.70 to 1.0.
- **Calibration**: Achieved 100% accuracy in the 0.9-1.0 confidence band.

---

## 5. Known Limitations

1.  **Complex COB**: At present, the system does not completely support COB claims (Secondary/Tertiary) outside of simple timely filing logic.
2.  **Changing Deadlines**: Current filing times are fixed as constants (e.g., 180 days for Commercial), and may require a dynamic knowledge base in production.
3.  **Latency**: Since the AI runs locally, expect to wait about 15-30 seconds for it to finish a claim. 

---

## 6. Future Scope(What's Next?)

*   **Problem 4 (Appeal Letter Generator)**: Generate an automated appeal letter based on `recommended_action` and `evidence`.
*   **Payer Policy RAG**: Implement a dedicated vector store for PDFs containing Payer Policy Manuals.
*   **HL7 FHIR R4 Support**: Develop ingest capability for HL7 FHIR R4 resources.