import pytest
import os
import json
from src.ingestion.loader import Loader
from src.ingestion.carc_lookup import CarcLookup
from src.analysis.p1_root_cause import RootCauseAnalyzer
from src.llm.client import OllamaClient

# Mock the LLM client to avoid hitting Ollama during fast tests
class MockLLM:
    def generate_sync(self, prompt, **kwargs):
        # We simulate a deepseek-r1 output with <think> tags
        return """<think>
I need to output the JSON schema.
</think>
```json
{
  "root_cause": "The claim was submitted after the 180 day commercial limit.",
  "carc_interpretation": "Time limit expired.",
  "recoverability": "not_recoverable",
  "confidence": 1.0,
  "evidence": ["Filing gap: 278 days"],
  "recommended_action": "Write off."
}
```"""

def test_p1_end_to_end_smoke(monkeypatch):
    # Mock LLM
    monkeypatch.setattr("src.analysis.p1_root_cause.llm_client", MockLLM())
    
    loader = Loader()
    lookup = CarcLookup("dummy.csv")
    analyzer = RootCauseAnalyzer(lookup)
    
    # Analyze Claim A (CARC 29)
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "samples", "CLM-2026-00142.json")
    if not os.path.exists(sample_path):
        pytest.skip(f"Sample not found: {sample_path}")
        
    record = loader.parse_sample_json(sample_path)
    analysis = analyzer.analyze_claim(record, "No similar context.")
    
    assert analysis.claim_id == "CLM-2026-00142"
    assert analysis.recoverability == "not_recoverable" # Rule engine sets this
    assert "278 days" in analysis.evidence[0] # From rule engine
    assert analysis.confidence == 1.0
