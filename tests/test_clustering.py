import pytest
import os
import sqlite3
import numpy as np
from src.models import ClaimRecord, DenialAnalysis
from src.storage.claim_store import ClaimStore
from src.analysis.p3_clustering import DenialClusterer

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_claims.db"
    return str(db_file)

@pytest.fixture
def claim_store(temp_db):
    return ClaimStore(db_path=temp_db)

def test_claim_store_persistence(claim_store):
    claim = ClaimRecord(
        claim_id="T1", payer_name="PayerX", insurance_type="Medicaid",
        carc_code="16", procedure_code="99214", principal_diagnosis="Z01",
        claim_status="", claim_amount=150.0, claim_paid=0, received_date="",
        statement_begin="", carc_description="", adjustment_group="",
        adjustment_amount=0, procedure_modifier="", remark_codes="",
        service_date_from="", diag2="", prior_authorization="",
        delay_reason_code="", claim_frequency="", bill_prov_npi="",
        subscriber_id="", type_of_bill="", rend_prov_specialty=""
    )
    analysis = DenialAnalysis(
        claim_id="T1", root_cause="Test cause", carc_interpretation="Test",
        recoverability="recoverable", confidence=0.9, evidence=[], recommended_action="Resubmit"
    )
    
    claim_store.save_claim(claim)
    claim_store.save_analysis(analysis)
    
    combined = claim_store.get_all_denied_claims_with_analysis()
    assert len(combined) == 1
    assert combined[0]["claim"]["claim_id"] == "T1"
    assert combined[0]["analysis"]["recoverability"] == "recoverable"

# Mock LLM for clusterer
class MockLLMForClusterer:
    def generate_sync(self, prompt, **kwargs):
        return """<think>
Processing cluster...
</think>
```json
{
  "cluster_label": "Test Cluster",
  "dominant_denial_reason": "Mock reason",
  "recovery_opportunity": "Mock opportunity",
  "estimated_recovery_rate": 50,
  "recommended_bulk_action": "Mock action"
}
```"""

def test_clustering_logic(monkeypatch):
    monkeypatch.setattr("src.analysis.p3_clustering.llm_client", MockLLMForClusterer())
    
    # Create 5 mock items
    combined_data = []
    for i in range(5):
        combined_data.append({
            "claim": {"claim_id": f"C{i}", "carc_code": "29" if i < 3 else "16", 
                      "insurance_type": "Commercial", "adjustment_group": "CO", 
                      "claim_amount": 100.0, "payer_name": "Aetna", "procedure_code": "99213"},
            "analysis": {"recoverability": "recoverable" if i < 3 else "needs_review"}
        })
        
    clusterer = DenialClusterer(n_clusters=2)
    insights = clusterer.cluster_denials(combined_data)
    
    assert len(insights) <= 2
    
    # Verify aggregate logic works
    total_claims = sum(i.num_claims for i in insights)
    assert total_claims == 5
    
    # Check if the mock LLM was parsed
    assert insights[0].summary_json["cluster_label"] == "Test Cluster"
