import json
import os
import random
from datetime import datetime, timedelta

def random_date(start_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(2026, 3, 1)
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate_claim(claim_idx):
    claim_id = f"SYN-2026-{claim_idx:04d}"
    
    ins_type = random.choices(
        ["Commercial", "Medicare", "Medicaid"], 
        weights=[0.60, 0.25, 0.15], k=1
    )[0]
    
    # Tier 1 CARC logic distribution
    scenarios = [
        {"carc": "4", "outcome": "needs_review", "desc": "Modifier mismatch"},
        {"carc": "16", "outcome": "needs_review", "desc": "Missing info"},
        {"carc": "18", "outcome": "not_recoverable", "desc": "Duplicate with ref"},
        {"carc": "18", "outcome": "needs_review", "desc": "Duplicate no ref"},
        {"carc": "29", "outcome": "recoverable", "desc": "Timely filing - good"},
        {"carc": "29", "outcome": "not_recoverable", "desc": "Timely filing - bad"},
        {"carc": "50", "outcome": "needs_review", "desc": "Medical necessity auth"},
        {"carc": "96", "outcome": "not_recoverable", "desc": "Non-covered with remark"},
        {"carc": "97", "outcome": "needs_review", "desc": "Bundled"},
        {"carc": "197", "outcome": "not_recoverable", "desc": "No prior auth"},
        {"carc": "252", "outcome": "recoverable", "desc": "Attachment needed"}
    ]
    
    scenario = random.choice(scenarios)
    
    # Base Dates
    service_date = random_date()
    
    # Modify dates for CARC 29
    if scenario["carc"] == "29":
        if scenario["outcome"] == "recoverable":
            received_date = service_date + timedelta(days=random.randint(10, 100))
        else:
            limit = 365 if ins_type == "Medicare" else 180
            received_date = service_date + timedelta(days=random.randint(limit + 10, limit + 100))
    else:
        received_date = service_date + timedelta(days=random.randint(10, 60))
        
    amount = round(random.uniform(100.0, 5000.0), 2)
    
    # 835 Data
    d835 = {
        "pc_ClaimID": claim_id,
        "pc_ClaimStatus": "4",
        "pc_ClaimAmount": amount,
        "pc_ClaimPaid": 0.0,
        "pc_InsuranceType": ins_type,
        "pc_ReceivedDate": received_date.strftime("%Y-%m-%d"),
        "pc_StatementBegin": service_date.strftime("%Y-%m-%d"),
        "pcla_AdjustmentGroup": "CO",
        "pcla_AdjustmentReason": scenario["carc"],
        "pcla_AdjustmentAmount": amount,
        "pcl_ProcedureCode": random.choice(["99213", "99214", "72148", "27447"]),
        "pcl_ProcedureModifier1": "25" if scenario["carc"] != "4" else "",
        "pcl_RemarkCodes": "N20" if scenario["carc"] in ("16", "96") else ""
    }
    
    # 837 Data
    d837 = {
        "ec_ClaimNo": claim_id,
        "ec_PayerName": f"{ins_type} Payer {random.randint(1,5)}",
        "ec_InsuranceType": ins_type,
        "ec_ServiceDateFrom": service_date.strftime("%Y-%m-%d"),
        "ec_PrincipalDiagnosis": random.choice(["J06.9", "M54.5", "M17.11", "E11.9"]),
        "ec_BillProvNPI": "1234567890",
        "ec_ClaimFrequency": "1",
        "ec_SubscriberID": f"SUB{random.randint(1000,9999)}"
    }
    
    # Specific overrides based on CARC
    if scenario["carc"] == "18" and scenario["outcome"] == "not_recoverable":
        d835["pc_OrigRefNo"] = f"REF-{random.randint(1000,9999)}"
    if scenario["carc"] == "16":
        d837["ec_PrincipalDiagnosis"] = "" # Missing field
    if scenario["carc"] in ("50", "197"):
        d837["ec_PriorAuthorization"] = ""
    else:
        d837["ec_PriorAuthorization"] = f"AUTH-{random.randint(100,999)}" if random.random() > 0.5 else ""
        
    ground_truth = {
        "claim_id": claim_id,
        "recoverability": scenario["outcome"],
        "carc": scenario["carc"],
        "description": scenario["desc"]
    }
    
    return {"835": d835, "837": d837, "ground_truth": ground_truth}

def seed_database(claims):
    from src.ingestion.loader import Loader
    from src.models import DenialAnalysis
    from src.storage.claim_vector_store import TurboStore
    from src.analysis.p2_pattern_match import PatternMatcher
    
    loader = Loader()
    store = TurboStore()
    matcher = PatternMatcher(store=store)
    
    print(f"Seeding TurboStore with {len(claims)} claims...")
    for c in claims:
        # Load via the actual loader
        record = loader._to_claim_record(c["835"], c["837"])
        
        # Create a mock analysis using ground truth
        mock_analysis = DenialAnalysis(
            claim_id=record.claim_id,
            root_cause=f"Mock root cause for {c['ground_truth']['description']}",
            carc_interpretation="Mock interpretation",
            recoverability=c['ground_truth']['recoverability'],
            confidence=1.0,
            evidence=[]
        )
        
        # Embed and upsert
        embedding = matcher.embed_claim(record)
        store.upsert_claim(record, mock_analysis, embedding)
        
    print("Seeding complete.")

if __name__ == "__main__":
    random.seed(42) # Deterministic for evaluation
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    
    num_claims = 35
    all_claims = [generate_claim(i) for i in range(1, num_claims + 1)]
    
    # Separate outputs
    claims_file = os.path.join(output_dir, "claims.json")
    truth_file = os.path.join(output_dir, "ground_truth.json")
    
    with open(claims_file, 'w', encoding='utf-8') as f:
        json.dump(all_claims, f, indent=2)
        
    ground_truths = {c["ground_truth"]["claim_id"]: c["ground_truth"] for c in all_claims}
    with open(truth_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truths, f, indent=2)
        
    print(f"Generated {num_claims} synthetic claims to {claims_file}")
    
    # Seed TurboStore
    seed_database(all_claims)
