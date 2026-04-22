# Synthetic Dataset Documentation

This directory contains a generated dataset of **35 denied insurance claims** used to train the Pattern Matcher and validate the Root Cause Engine.

## Edge Cases Covered

1.  **Borderline Timely Filing (CARC 29)**:
    - Commercial claims submitted on Day 179 (Pass) vs Day 181 (Fail).
    - Medicare claims submitted on Day 360 (Pass).
2.  **Replacement Claims (CARC 18)**:
    - Claims with `ClaimFrequency = 7` (Replacement) which are flagged as duplicates by payers but are actually recoverable corrections.
3.  **Missing Authorization (CARC 197)**:
    - Claims with explicitly null `prior_authorization` fields.
4.  **Incomplete Data (CARC 16)**:
    - Claims missing Rendering NPI, Subscriber IDs, and Procedure Modifiers to test the field-level detection logic.
5.  **Non-Covered Services (CARC 96)**:
    - Claims with and without the `N130` remark code to test the confidence calibration logic.

## Ground Truth Construction
Every claim in `claims.json` was generated with a corresponding "Reasoning Intent" to ensure the deterministic rule engine and the LLM could be measured against objective billing facts.
