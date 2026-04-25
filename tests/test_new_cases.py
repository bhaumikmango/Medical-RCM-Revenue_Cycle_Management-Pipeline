import pytest
import datetime
from src.models import ClaimRecord
from src.analysis.p1_root_cause import RootCauseAnalyzer
from src.ingestion.carc_lookup import CarcLookup

@pytest.fixture
def analyzer():
    # Use a real CARC reference if possible, but dummy is fine for logic testing
    lookup = CarcLookup("data/raw/carc_reference.csv")
    return RootCauseAnalyzer(lookup)

def create_claim(**kwargs):
    """Helper to create a ClaimRecord with sensible defaults."""
    defaults = {
        "claim_id": "TEST-001",
        "payer_name": "Anthem",
        "claim_status": "4",
        "claim_amount": 500.0,
        "claim_paid": 0.0,
        "insurance_type": "Commercial",
        "received_date": "2026-05-01",
        "statement_begin": "2026-01-01",
        "carc_code": "29",
        "carc_description": "Timely Filing",
        "adjustment_group": "CO",
        "adjustment_amount": 500.0,
        "procedure_code": "99214",
        "procedure_modifier": "",
        "remark_codes": "",
        "service_date_from": "2026-01-01",
        "principal_diagnosis": "M54.5",
        "diag2": "",
        "prior_authorization": "",
        "delay_reason_code": "",
        "claim_frequency": "1",
        "bill_prov_npi": "1234567890",
        "subscriber_id": "SUB123",
        "type_of_bill": "111",
        "rend_prov_specialty": "Internal Medicine",
        "original_ref": "",
        "patient_relationship": "01" # Self
    }
    defaults.update(kwargs)
    return ClaimRecord(**defaults)

# --- SCENARIO: CARC 29 with COB (Secondary Claim) ---
def test_carc29_secondary_claim_routing(analyzer):
    """
    Scenario: CARC 29 with COB (Secondary Claim)
    Given a claim with CARC 29 (Timely Filing)
    And the patient relationship is '18' (Self - Secondary)
    When the rule engine analyzes the claim
    Then it should return 'needs_review' even if the filing limit is exceeded
    And the confidence should be 0.7
    """
    # 121 days elapsed (Commercial limit is 180, so it's technically fine, 
    # but let's test a fail case like 200 days)
    claim = create_claim(
        received_date="2026-08-01", 
        service_date_from="2026-01-01", # 212 days
        patient_relationship="18" # Self/Secondary
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.7
    assert "Secondary claim (COB)" in flags["evidence"][0]

# --- SCENARIO: CARC 197 with Delay Reason Code ---
def test_carc197_with_delay_reason_code(analyzer):
    """
    Scenario: CARC 197 with Delay Reason Code
    Given a claim with CARC 197 (Auth Missing)
    And a delay reason code '1' is present
    When the rule engine analyzes the claim
    Then it should return 'needs_review' instead of 'not_recoverable'
    And the evidence should cite the delay reason code.
    """
    claim = create_claim(
        carc_code="197",
        prior_authorization="",
        delay_reason_code="1"
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "needs_review"
    assert flags["confidence"] == 0.7
    assert "Delay reason code 1 present" in flags["evidence"][0]

# --- SCENARIO: Invalid Date Format Handling ---
def test_invalid_date_format_handling(analyzer):
    """
    Scenario: Invalid Date Format Handling
    Given a claim with an invalid date string '01-01-2026' (wrong format)
    When the rule engine attempts to parse the date for CARC 29
    Then it should handle the error gracefully
    And return 'needs_review' with a specific evidence message.
    """
    claim = create_claim(
        carc_code="29",
        received_date="05-01-2026", # Wrong format (should be YYYY-MM-DD)
        service_date_from="2026-01-01"
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "needs_review"
    assert "Invalid dates" in flags["evidence"][0]

# --- SCENARIO: Missing Critical Fields (Negative/Severity) ---
def test_carc16_missing_critical_fields_severity(analyzer):
    """
    Scenario: Missing Critical Fields (Negative)
    Given a claim with CARC 16 (Missing Info)
    And critical fields ec_BillProvNPI and ec_SubscriberID are missing
    When the rule engine analyzes the claim
    Then it should classify the severity as 'major'
    And return 'needs_review'.
    """
    claim = create_claim(
        carc_code="16",
        bill_prov_npi="", # Critical
        subscriber_id="", # Critical
        principal_diagnosis="M54.5" # Present
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "needs_review"
    assert flags["missing_severity"] == "major"
    assert "ec_BillProvNPI" in flags["missing_fields"]
    assert "ec_SubscriberID" in flags["missing_fields"]

# --- SCENARIO: CARC 96 with Multiple Remark Codes ---
def test_carc96_multiple_remark_codes(analyzer):
    """
    Scenario: CARC 96 with Multiple Remark Codes
    Given a claim with CARC 96 (Non-covered)
    And multiple remark codes 'MA01, N20, N130'
    When the rule engine analyzes the claim
    Then it should identify the first non-covered remark (N20)
    And lock to 'not_recoverable' with 0.9 confidence.
    """
    claim = create_claim(
        carc_code="96",
        remark_codes="MA01, N20, N130"
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "not_recoverable"
    assert flags["confidence"] == 0.9
    assert any(code in flags["evidence"][0] for code in ["N20", "N130", "N95"])

# --- SCENARIO: CARC 18 (Duplicate) with Void/Correction Frequency ---
def test_carc18_correction_frequency(analyzer):
    """
    Scenario: CARC 18 (Duplicate) with Void/Correction Frequency
    Given a claim with CARC 18
    And a claim frequency of '7' (Replacement)
    When the rule engine analyzes the claim
    Then it should be 'recoverable'
    And confidence should be 1.0.
    """
    claim = create_claim(
        carc_code="18",
        claim_frequency="7"
    )
    
    flags = analyzer.run_rule_engine(claim)
    
    assert flags["recoverability"] == "recoverable"
    assert flags["confidence"] == 1.0
    assert "Correction/Void" in flags["evidence"][0]
