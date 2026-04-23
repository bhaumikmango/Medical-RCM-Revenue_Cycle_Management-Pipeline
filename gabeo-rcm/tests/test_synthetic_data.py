import json
import os
import pytest

def test_synthetic_files_exist():
    output_dir = "data/synthetic"
    assert os.path.exists(os.path.join(output_dir, "claims.json"))
    assert os.path.exists(os.path.join(output_dir, "ground_truth.json"))

def test_synthetic_data_integrity():
    with open("data/synthetic/claims.json", 'r', encoding='utf-8') as f:
        claims = json.load(f)
    with open("data/synthetic/ground_truth.json", 'r', encoding='utf-8') as f:
        truth = json.load(f)
        
    assert len(claims) >= 39 # 35 base + 4 edge cases
    
    # Check for specific edge case IDs
    target_ids = {"SYN-2026-COB-001", "SYN-2026-CORR-001", "SYN-2026-AUTH-001", "SYN-2026-MISS-001"}
    found_ids = {c["835"]["pc_ClaimID"] for c in claims}
    
    for tid in target_ids:
        assert tid in found_ids
        assert tid in truth

def test_claim_structure():
    with open("data/synthetic/claims.json", 'r', encoding='utf-8') as f:
        claims = json.load(f)
        
    for c in claims:
        assert "835" in c
        assert "837" in c
        assert "pc_ClaimID" in c["835"]
        assert "ec_ClaimNo" in c["837"]
        assert c["835"]["pc_ClaimID"] == c["837"]["ec_ClaimNo"]
