import pytest
import json
import os
from unittest.mock import MagicMock, patch
from scripts.evaluate import Evaluator

def test_evaluation_math():
    """
    Verifies the precision/recall math and confidence banding logic without running LLM.
    """
    evaluator = Evaluator()
    
    # Mock pipeline and store
    mock_pipeline = MagicMock()
    evaluator.pipeline = mock_pipeline
    
    # Setup mock results (3 claims)
    mock_results = [
        {
            "claim": {"claim_id": "SYN-1", "carc_code": "29"},
            "analysis": {"recoverability": "recoverable", "confidence": 0.95}
        },
        {
            "claim": {"claim_id": "SYN-2", "carc_code": "253"},
            "analysis": {"recoverability": "not_recoverable", "confidence": 0.8}
        },
        {
            "claim": {"claim_id": "SYN-3", "carc_code": "252"},
            "analysis": {"recoverability": "recoverable", "confidence": 0.4}
        }
    ]
    mock_pipeline.claim_store.get_all_denied_claims_with_analysis.return_value = mock_results
    
    # Setup mock ground truth
    ground_truth = {
        "SYN-1": {"recoverability": "recoverable"},
        "SYN-2": {"recoverability": "not_recoverable"},
        "SYN-3": {"recoverability": "not_recoverable"} # Intentional mismatch
    }
    
    # We patch json.load and open to return our mock data
    with patch("builtins.open", MagicMock()):
        with patch("json.load", side_effect=[[{}], ground_truth]):
            with patch("os.path.exists", return_value=True):
                metrics = evaluator.run_evaluation("fake_claims.json", "fake_gt.json")
                
                # Assertions
                assert metrics["total_processed"] == 3
                assert metrics["avg_confidence"] == round((0.95 + 0.8 + 0.4) / 3, 2)
                
                # Check classification report existence
                assert "classification_report" in metrics
                # SYN-1 is correct, SYN-2 is correct, SYN-3 is wrong.
                # Accuracy should be 2/3 = 0.666...
                assert metrics["classification_report"]["accuracy"] == pytest.approx(2/3, 0.01)
                
                # Check calibration bands
                # SYN-1 (0.95) -> 0.9-1.0 band (Correct)
                # SYN-2 (0.8) -> 0.7-0.9 band (Correct)
                # SYN-3 (0.4) -> 0.0-0.5 band (Wrong)
                assert metrics["confidence_calibration"]["0.9-1.0"]["accuracy"] == 1.0
                assert metrics["confidence_calibration"]["0.0-0.5"]["accuracy"] == 0.0
                
                # Deterministic check (Rule accuracy)
                # CARC 253 (SYN-2) -> not_recoverable (Pass)
                # CARC 252 (SYN-3) -> recoverable (Pass)
                # No logic errors here
                assert metrics["deterministic_rule_accuracy"] == 100.0
