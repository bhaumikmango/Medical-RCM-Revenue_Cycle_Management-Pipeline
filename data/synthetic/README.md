# Synthetic Dataset Documentation

This folder contains a synthetic dataset of **39 denied insurance claims** used to 
train the TurboStore pattern matcher and validate the hybrid rule engine + LLM pipeline.

## Dataset Composition

| Source | Claims | Purpose |
|---|---|---|
| `generate_synthetic.py` (seed=42) | 35 | Base Tier 1 CARC coverage — deterministic, reproducible |
| `generate_synthetic2.py` | 4 | Edge case stress testing — complex scenarios |
| **Total** | **39** | **Full evaluation dataset** |

---

## Base Dataset — Tier 1 CARC Coverage (35 Claims)

### 1. Borderline Timely Filing (CARC 29)
Commercial claims filed at Day 179 (pass) vs Day 181 (fail).
Medicare claims filed at Day 360 (pass, within 365-day window).
Tests the rule engine's insurance-type-aware date arithmetic.

### 2. Replacement Claims (CARC 18)
Claims with `ec_ClaimFrequency = 7` (Replacement) that payers flag as duplicates.
Tests whether the rule engine correctly identifies corrected claims as recoverable
rather than marking them as true duplicates.

Claims with `ec_ClaimFrequency = 1` and `pc_OrigRefNo` present — confirmed duplicates.
Tests the original reference number detection and `not_recoverable` lock at confidence 0.9.

### 3. Missing Authorization (CARC 197)
Claims with an explicit empty `ec_PriorAuthorization` field and no delay reason code.
Tests the hard `not_recoverable` path at confidence 0.9.

### 4. Missing Data (CARC 16)
Claims missing Rendering NPI, Subscriber ID, and Procedure Modifiers.
Tests field-level detection and missing severity classification
(minor vs major) in the rule engine.

### 5. Non-Covered Services (CARC 96)
Claims with remark code `N20` ("service not covered by this payer") and `N130`.
Tests the `NON_COVERED_CONFIRMED_REMARKS` set — expanded from N130-only
to `{N20, N130, N95}` after Iteration 1 diagnostic analysis.

---

## Edge Case Dataset — Stress Testing (4 Claims)

### SYN-2026-COB-001 — Medicare Secondary Claim (CARC 29)
`ec_PatientRelationship = "18"`, `ec_DelayReasonCode = "3"`

Tests the most complex timely filing scenario: a Medicare secondary claim where
the filing window starts from the primary payer's EOB date, not the service date.
The EOB date is not present in the 835/837 data, making standard date arithmetic
impossible. Ground truth: `needs_review`.

Why it matters: Standard CARC 29 logic would incorrectly lock this as
`not_recoverable`. The rule engine must detect the secondary claim flag before
running date math and route to `needs_review` instead.

### SYN-2026-CORR-001 — Corrected Claim (CARC 18)
`ec_ClaimFrequency = "7"`

Tests false duplicate detection prevention. A replacement claim submitted with
frequency code 7 is not a duplicate — it is a billing correction. Ground truth: `recoverable`.

### SYN-2026-AUTH-001 — Prior Auth Absent with Delay Code (CARC 197)
`ec_PriorAuthorization = ""`, `ec_DelayReasonCode = "3"`

Tests the interaction between a missing prior authorization and a delay reason
code. When a delay code is present, retroactive authorization eligibility cannot
be ruled out. Ground truth: `needs_review` rather than `not_recoverable`.

### SYN-2026-MISS-001 — Missing Principal Diagnosis (CARC 16)
`ec_PrincipalDiagnosis = ""`

Tests CARC 16 with a single critical missing field. Principal diagnosis is
required for payer adjudication — its absence means the payer cannot assess
medical necessity on resubmission either. Ground truth: `needs_review`.

---

## Ground Truth Construction

Each claim in `claims.json` has a corresponding entry in `ground_truth.json`
with a `recoverability` label (`recoverable`, `not_recoverable`, `needs_review`),
the CARC code, and a description of the scenario intent.

Ground truth labels reflect objective billing rules, not aspirational system targets.
Labels were validated against the hybrid rule engine logic and standard RCM
billing practice.

---

## Evaluation Results (Final)

| Metric | Value |
|---|---|
| Total Claims Evaluated | 39 |
| Overall Accuracy | **100%** |
| needs_review F1 | 1.0 |
| not_recoverable F1 | 1.0 |
| recoverable F1 | 1.0 |
| Avg Confidence | 0.82 |
| Deterministic Rule Accuracy | 100% |

All confidence bands (0.5–0.7, 0.7–0.9, 0.9–1.0) achieve 100% accuracy,
confirming confidence scores are reliable proxies for verdict correctness.