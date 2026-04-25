import pytest
import datetime
from src.models import ClaimRecord
from src.analysis.p1_root_cause import RootCauseAnalyzer
from src.ingestion.carc_lookup import CarcLookup

@pytest.fixture
def analyzer():
    lookup = CarcLookup("dummy.csv")
    return RootCauseAnalyzer(lookup)

def dummy_claim(carc, **kwargs):
    defaults = {
        "claim_id": "T1", "payer_name": "Test", "claim_status": "4",
        "claim_amount": 100.0, "claim_paid": 0.0, "insurance_type": "Commercial",
        "received_date": "2026-01-01", "statement_begin": "",
        "carc_code": carc, "carc_description": "", "adjustment_group": "CO",
        "adjustment_amount": 100.0, "procedure_code": "99213", "procedure_modifier": "",
        "remark_codes": "", "service_date_from": "2026-01-01", "principal_diagnosis": "A00",
        "diag2": "", "prior_authorization": "", "delay_reason_code": "",
        "claim_frequency": "1", "bill_prov_npi": "123", "subscriber_id": "123",
        "type_of_bill": "", "rend_prov_specialty": ""
    }
    defaults.update(kwargs)
    return ClaimRecord(**defaults)

def test_rule_engine_carc_29_commercial_pass(analyzer):
    claim = dummy_claim("29", received_date="2026-06-01", service_date_from="2026-01-01") # 151 days
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "recoverable"
    assert flags["confidence"] == 1.0

def test_rule_engine_carc_29_commercial_fail(analyzer):
    claim = dummy_claim("29", received_date="2026-08-01", service_date_from="2026-01-01") # 212 days
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "not_recoverable"
    assert flags["confidence"] == 1.0

def test_rule_engine_carc_29_medicare_pass(analyzer):
    claim = dummy_claim("29", received_date="2026-11-01", service_date_from="2026-01-01", insurance_type="Medicare") # 304 days
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "recoverable"
    assert flags["confidence"] == 1.0

def test_rule_engine_carc_252_fixed(analyzer):
    claim = dummy_claim("252")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "recoverable"
    assert flags["confidence"] == 0.9

def test_rule_engine_carc_18_no_ref(analyzer):
    claim = dummy_claim("18")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.85

def test_rule_engine_carc_50_logic(analyzer):
    claim = dummy_claim("50")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.75

def test_rule_engine_carc_97_logic(analyzer):
    claim = dummy_claim("97")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.6

def test_rule_engine_carc_96_remark_absent(analyzer):
    claim = dummy_claim("96", remark_codes="")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.5

def test_rule_engine_carc_96_remark_present(analyzer):
    claim = dummy_claim("96", remark_codes="N130")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "not_recoverable"
    assert flags["confidence"] == 0.9

def test_rule_engine_carc_16_logic(analyzer):
    claim = dummy_claim("16", principal_diagnosis="") # missing dx
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.9

def test_carc18_with_original_ref_locks_not_recoverable(analyzer):
    claim = dummy_claim("18", original_ref="REF-5010")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "not_recoverable"
    assert flags["confidence"] == 0.9

def test_carc18_without_original_ref_stays_needs_review(analyzer):
    claim = dummy_claim("18", original_ref="")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.85

def test_carc96_n20_remark_locks_not_recoverable(analyzer):
    claim = dummy_claim("96", remark_codes="N20")
    flags = analyzer.run_rule_engine(claim)
    assert flags["recoverability"] == "not_recoverable"
    assert flags["confidence"] == 0.9
