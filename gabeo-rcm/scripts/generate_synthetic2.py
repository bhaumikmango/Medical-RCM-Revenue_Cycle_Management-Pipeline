import json
import os
import random
from datetime import datetime, timedelta

def random_date(start_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(2026, 3, 1)
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate_edge_cases():
    edge_cases = []
    
    # 1. Medicare Secondary (COB)
    # Scenario: Primary paid $50, Medicare is secondary. CARC 29 (Timely) but it's okay because primary EOB was recent.
    # Actually, let's do CARC 29 with Medicare Secondary.
    cid1 = "SYN-2026-COB-001"
    svc_date = datetime(2025, 6, 1)
    # Received 400 days later (normally too late), but secondary.
    recv_date = svc_date + timedelta(days=400)
    
    edge_cases.append({
        "835": {
            "pc_ClaimID": cid1,
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 200.0,
            "pc_ClaimPaid": 0.0,
            "pc_InsuranceType": "Medicare",
            "pc_ReceivedDate": recv_date.strftime("%Y-%m-%d"),
            "pc_StatementBegin": svc_date.strftime("%Y-%m-%d"),
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "29",
            "pcla_AdjustmentAmount": 200.0,
            "pcl_ProcedureCode": "99214",
            "pcl_ProcedureModifier1": "25",
            "pcl_RemarkCodes": "N20"
        },
        "837": {
            "ec_ClaimNo": cid1,
            "ec_PayerName": "Medicare Part B",
            "ec_InsuranceType": "Medicare",
            "ec_ServiceDateFrom": svc_date.strftime("%Y-%m-%d"),
            "ec_PrincipalDiagnosis": "M54.5",
            "ec_BillProvNPI": "1234567890",
            "ec_ClaimFrequency": "1",
            "ec_SubscriberID": "SUB1001",
            "ec_PatientRelationship": "18" # Self
        },
        "ground_truth": {
            "claim_id": cid1,
            "recoverability": "needs_review", # COB is complex
            "carc": "29",
            "description": "Medicare Secondary timely filing"
        }
    })

    # 2. Corrected Claim (Freq 7)
    # Scenario: CARC 18 (Duplicate) but Freq 7 means it's a replacement.
    cid2 = "SYN-2026-CORR-001"
    edge_cases.append({
        "835": {
            "pc_ClaimID": cid2,
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 150.0,
            "pc_ClaimPaid": 0.0,
            "pc_InsuranceType": "Commercial",
            "pc_ReceivedDate": "2026-02-01",
            "pc_StatementBegin": "2026-01-01",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "18",
            "pcla_AdjustmentAmount": 150.0,
            "pcl_ProcedureCode": "99213",
            "pcl_ProcedureModifier1": "25",
            "pcl_RemarkCodes": ""
        },
        "837": {
            "ec_ClaimNo": cid2,
            "ec_PayerName": "Aetna",
            "ec_InsuranceType": "Commercial",
            "ec_ServiceDateFrom": "2026-01-01",
            "ec_PrincipalDiagnosis": "J06.9",
            "ec_BillProvNPI": "1234567890",
            "ec_ClaimFrequency": "7", # Corrected
            "ec_SubscriberID": "SUB2002"
        },
        "ground_truth": {
            "claim_id": cid2,
            "recoverability": "recoverable",
            "carc": "18",
            "description": "Corrected claim frequency 7"
        }
    })

    # 3. Auth Ambiguity (CARC 197 + Delay Code)
    # Scenario: No prior auth, but we have a delay reason code (maybe an emergency).
    cid3 = "SYN-2026-AUTH-001"
    edge_cases.append({
        "835": {
            "pc_ClaimID": cid3,
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 3000.0,
            "pc_ClaimPaid": 0.0,
            "pc_InsuranceType": "Commercial",
            "pc_ReceivedDate": "2026-03-01",
            "pc_StatementBegin": "2026-02-01",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "197",
            "pcla_AdjustmentAmount": 3000.0,
            "pcl_ProcedureCode": "27447",
            "pcl_ProcedureModifier1": "",
            "pcl_RemarkCodes": ""
        },
        "837": {
            "ec_ClaimNo": cid3,
            "ec_PayerName": "Cigna",
            "ec_InsuranceType": "Commercial",
            "ec_ServiceDateFrom": "2026-02-01",
            "ec_PrincipalDiagnosis": "M17.11",
            "ec_BillProvNPI": "1234567890",
            "ec_ClaimFrequency": "1",
            "ec_SubscriberID": "SUB3003",
            "ec_DelayReasonCode": "1" # Exceptional circumstances
        },
        "ground_truth": {
            "claim_id": cid3,
            "recoverability": "needs_review",
            "carc": "197",
            "description": "No auth with delay reason"
        }
    })

    # 4. Simple Omission (Missing DX)
    # Scenario: CARC 16 (Missing info) and indeed ec_PrincipalDiagnosis is empty.
    cid4 = "SYN-2026-MISS-001"
    edge_cases.append({
        "835": {
            "pc_ClaimID": cid4,
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 100.0,
            "pc_ClaimPaid": 0.0,
            "pc_InsuranceType": "Medicaid",
            "pc_ReceivedDate": "2026-03-10",
            "pc_StatementBegin": "2026-03-01",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "16",
            "pcla_AdjustmentAmount": 100.0,
            "pcl_ProcedureCode": "99213",
            "pcl_ProcedureModifier1": "",
            "pcl_RemarkCodes": "M76"
        },
        "837": {
            "ec_ClaimNo": cid4,
            "ec_PayerName": "Medicaid State",
            "ec_InsuranceType": "Medicaid",
            "ec_ServiceDateFrom": "2026-03-01",
            "ec_PrincipalDiagnosis": "", # Omitted
            "ec_BillProvNPI": "1234567890",
            "ec_ClaimFrequency": "1",
            "ec_SubscriberID": "SUB4004"
        },
        "ground_truth": {
            "claim_id": cid4,
            "recoverability": "recoverable",
            "carc": "16",
            "description": "Missing principal diagnosis"
        }
    })

    return edge_cases

def merge_and_seed(new_claims):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
    claims_file = os.path.join(output_dir, "claims.json")
    truth_file = os.path.join(output_dir, "ground_truth.json")

    # Read existing
    with open(claims_file, 'r', encoding='utf-8') as f:
        existing_claims = json.load(f)
    with open(truth_file, 'r', encoding='utf-8') as f:
        existing_truth = json.load(f)

    # Add new
    existing_claims.extend(new_claims)
    for c in new_claims:
        existing_truth[c["ground_truth"]["claim_id"]] = c["ground_truth"]

    # Save
    with open(claims_file, 'w', encoding='utf-8') as f:
        json.dump(existing_claims, f, indent=2)
    with open(truth_file, 'w', encoding='utf-8') as f:
        json.dump(existing_truth, f, indent=2)

    print(f"Added {len(new_claims)} edge cases to {claims_file}")

    # Seed TurboStore (re-use logic from generate_synthetic.py)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from scripts.generate_synthetic import seed_database
    seed_database(new_claims)

if __name__ == "__main__":
    new_cases = generate_edge_cases()
    merge_and_seed(new_cases)
