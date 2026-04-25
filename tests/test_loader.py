import pytest
import os
import json
import pandas as pd
from src.ingestion.loader import Loader
from src.ingestion.validator import Validator
from src.ingestion.carc_lookup import CarcLookup

@pytest.fixture
def loader():
    return Loader()

@pytest.fixture
def validator():
    return Validator()

def test_parse_sample_json(loader):
    # Create a dummy JSON
    dummy = {
        "835": {"pc_ClaimID": "TEST-01", "pc_ClaimAmount": 100.0, "pcla_AdjustmentReason": "29"},
        "837": {"ec_ClaimNo": "TEST-01", "ec_ServiceDateFrom": "2026-01-01"}
    }
    with open("test_dummy.json", "w") as f:
        json.dump(dummy, f)
        
    try:
        record = loader.parse_sample_json("test_dummy.json")
        assert record is not None
        assert record.claim_id == "TEST-01"
        assert record.claim_amount == 100.0
        assert record.carc_code == "29"
        assert record.service_date_from == "2026-01-01"
    finally:
        if os.path.exists("test_dummy.json"):
            os.remove("test_dummy.json")

def test_validator_missing_fields(validator):
    bad_record = {"claim_id": "TEST-02"} # Missing many required
    res = validator.validate(bad_record)
    assert not res["is_valid"]
    assert len(res["errors"]) > 0

def test_validator_carc_16_remark_flag(validator):
    rec_no_remark = {"claim_id": "T1", "carc_code": "16", "remark_codes": "", "claim_amount": 1, "received_date": "d", "service_date_from": "d", "insurance_type": "i"}
    res = validator.validate(rec_no_remark)
    assert "MISSING_REMARK_CODE" in res["warnings"]
    
    rec_with_remark = {"claim_id": "T1", "carc_code": "16", "remark_codes": "N20", "claim_amount": 1, "received_date": "d", "service_date_from": "d", "insurance_type": "i"}
    res2 = validator.validate(rec_with_remark)
    assert "MISSING_REMARK_CODE" not in res2["warnings"]

def test_join_claims_from_csv(loader):
    df_835 = pd.DataFrame([{"pc_ClaimID": "CLM-X", "pc_ClaimAmount": 50.0}])
    df_837 = pd.DataFrame([{"ec_ClaimNo": "CLM-X", "ec_ServiceDateFrom": "2026-01-01"}])
    
    records = loader.join_claims_from_csv(df_835, df_837)
    assert len(records) == 1
    assert records[0].claim_id == "CLM-X"
    assert records[0].claim_amount == 50.0
    assert records[0].service_date_from == "2026-01-01"

def test_carc_lookup():
    lookup = CarcLookup("dummy_path_that_doesnt_exist.csv") # Should load fallback
    desc = lookup.get("29")
    assert "time limit" in desc["description"].lower()
    assert desc["status"] == "active"
