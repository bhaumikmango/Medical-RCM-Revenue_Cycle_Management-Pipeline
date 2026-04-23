import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from src.models import ClaimRecord, DenialAnalysis
from src.analysis.appeal_writer import generate_appeal_letter

def run_async(coro):
    return asyncio.run(coro)

def test_generate_appeal_letter_formatting():
    """
    Verifies that the AppealWriter correctly formats the prompt and calls the LLM.
    """
    # 1. Setup Mock Data
    claim = ClaimRecord(
        claim_id="TEST-101",
        payer_name="Test Payer",
        claim_status="4",
        claim_amount=100.0,
        claim_paid=0.0,
        insurance_type="Commercial",
        received_date="2026-01-10",
        statement_begin="2026-01-01",
        carc_code="16",
        carc_description="Missing Info",
        adjustment_group="PR",
        adjustment_amount=100.0,
        procedure_code="99213",
        procedure_modifier="",
        remark_codes="N1",
        service_date_from="2026-01-01",
        principal_diagnosis="I10",
        diag2="",
        prior_authorization="",
        delay_reason_code="",
        claim_frequency="1",
        bill_prov_npi="1234567890",
        subscriber_id="SUB-001",
        type_of_bill="111",
        rend_prov_specialty="Internal Medicine"
    )
    
    analysis = DenialAnalysis(
        claim_id="TEST-101",
        root_cause="Missing Principal Diagnosis",
        carc_interpretation="The claim is missing clinical data.",
        recoverability="recoverable",
        confidence=0.9,
        evidence=["Field principal_diagnosis is empty"],
        recommended_action="Resubmit with diagnosis"
    )

    # 2. Mock the LLM Client
    mock_letter = "This is a mock appeal letter for TEST-101."
    
    # We use patch with AsyncMock and run it inside asyncio.run
    with patch("src.analysis.appeal_writer.llm_client.generate_async", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_letter
        
        # 3. Call the function using our helper
        result = run_async(generate_appeal_letter(analysis, claim))
        
        # 4. Assertions
        assert result == mock_letter
        mock_gen.assert_called_once()
        
        # Verify that the formatted evidence was passed to the prompt
        call_args = mock_gen.call_args[0][0]
        assert "1. Field principal_diagnosis is empty" in call_args
        assert "TEST-101" in call_args
        assert "Commercial" in call_args
        assert "I10" in call_args

def test_generate_appeal_letter_missing_file():
    """
    Tests handling of missing prompt template.
    """
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            run_async(generate_appeal_letter(None, None))
