# Synthetic Dataset Documentation

This folder contains a synthetic dataset of **35 denied insurance claims** to train the Pattern Matcher and validate the Root Cause Engine.

## Edge Cases Handled

1. **Borderline Timely Filing (CARC 29)**
    - Commercial claims filed Day 179 (Pass) vs Day 181 (Fail).
    Day 360 Medicare claims submitted (Pass).
2. **Replacement Claims (CARC 18)**
    - Claims with `ClaimFrequency = 7` (Replacement) that Payers identify as duplicates, but are in fact recoverable corrections.
3. **Missing Authorization (CARC 197)**
    - Claims with an explicit null value for the `prior_authorization` field.
4. **Missing Data (CARC 16):**
    - Claims with no Rendering NPI, Subscriber IDs and Procedure Modifiers for testing the field level detection logic.
5. **Non-Covered Services (CARC 96)**
    - Claims with and without the `N130` remark code to test the confidence calibration logic. 

## Constructing Ground Truth
Each claim in `claims.json` is associated with a "Reasoning Intent". This is to evaluate the deterministic rule engine and the LLM against objective facts of billing.