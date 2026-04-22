import pytest
import os
import shutil
import numpy as np
from src.storage.claim_vector_store import TurboStore
from src.storage.turbo_quant import TurboQuant
from src.analysis.p2_pattern_match import PatternMatcher
from src.models import ClaimRecord, DenialAnalysis

@pytest.fixture
def temp_store_dir(tmp_path):
    d = tmp_path / "vector_store"
    d.mkdir()
    return str(d)

@pytest.fixture
def store(temp_store_dir):
    return TurboStore(store_dir=temp_store_dir, dim=384)

@pytest.fixture
def matcher(store):
    return PatternMatcher(store=store)

def test_turbo_quant_logic():
    dim = 384
    tq = TurboQuant(dim=dim)
    vec = np.random.randn(1, dim).astype(np.float32)
    
    packed = tq.quantize(vec)
    assert packed.shape == (1, dim // 2)
    assert packed.dtype == np.uint8
    
    scores = tq.unbiased_inner_product(vec, packed)
    assert len(scores) == 1
    assert scores[0] > 0.9 # Should be very high similarity to self

def test_store_upsert_and_search(store):
    claim = ClaimRecord(
        claim_id="TEST-UPSERT", payer_name="Payer", claim_status="4",
        claim_amount=100.0, claim_paid=0.0, insurance_type="Commercial",
        received_date="2026-01-01", statement_begin="", carc_code="29",
        carc_description="", adjustment_group="CO", adjustment_amount=100.0,
        procedure_code="99213", procedure_modifier="", remark_codes="",
        service_date_from="2026-01-01", principal_diagnosis="A00", diag2="",
        prior_authorization="", delay_reason_code="", claim_frequency="1",
        bill_prov_npi="123", subscriber_id="123", type_of_bill="",
        rend_prov_specialty=""
    )
    analysis = DenialAnalysis(
        claim_id="TEST-UPSERT", root_cause="Test", carc_interpretation="Test",
        recoverability="recoverable", confidence=1.0, evidence=[]
    )
    
    embedding = np.random.randn(1, 384).astype(np.float32)
    store.upsert_claim(claim, analysis, embedding)
    
    # Verify files created
    assert os.path.exists(store.index_file)
    assert os.path.exists(store.meta_file)
    
    # Search
    results = store.search_similar(embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["claim_id"] == "TEST-UPSERT"
    assert results[0]["recoverability"] == "recoverable"

def test_pattern_matcher_text_builder(matcher):
    claim = ClaimRecord(
        claim_id="T1", payer_name="PayerX", insurance_type="Medicaid",
        carc_code="16", procedure_code="99214", principal_diagnosis="Z01",
        # other fields dummy
        claim_status="", claim_amount=0, claim_paid=0, received_date="",
        statement_begin="", carc_description="", adjustment_group="",
        adjustment_amount=0, procedure_modifier="", remark_codes="",
        service_date_from="", diag2="", prior_authorization="",
        delay_reason_code="", claim_frequency="", bill_prov_npi="",
        subscriber_id="", type_of_bill="", rend_prov_specialty=""
    )
    
    text = matcher._build_embedding_text(claim)
    assert "CARC:16" in text
    assert "payer:PayerX" in text
    assert "proc:99214" in text
    assert "dx:Z01" in text
    assert "ins:Medicaid" in text

def test_pattern_matcher_context_formatting(matcher):
    # Upsert a mock
    claim = ClaimRecord(
        claim_id="HIST-1", payer_name="P1", claim_status="4",
        claim_amount=100.0, claim_paid=0.0, insurance_type="Commercial",
        received_date="2026-01-01", statement_begin="", carc_code="29",
        carc_description="", adjustment_group="CO", adjustment_amount=100.0,
        procedure_code="99213", procedure_modifier="", remark_codes="",
        service_date_from="2026-01-01", principal_diagnosis="A00", diag2="",
        prior_authorization="", delay_reason_code="", claim_frequency="1",
        bill_prov_npi="123", subscriber_id="123", type_of_bill="",
        rend_prov_specialty=""
    )
    analysis = DenialAnalysis(
        claim_id="HIST-1", root_cause="Past limit", carc_interpretation="Test",
        recoverability="not_recoverable", confidence=1.0, evidence=[]
    )
    embedding = np.random.randn(1, 384).astype(np.float32)
    matcher.store.upsert_claim(claim, analysis, embedding)
    
    # Get context for same claim (should match)
    ctx = matcher.get_similar_context(claim, top_k=1)
    assert "HIST-1" in ctx
    assert "NOT_RECOVERABLE" in ctx
    assert "Past limit" in ctx
