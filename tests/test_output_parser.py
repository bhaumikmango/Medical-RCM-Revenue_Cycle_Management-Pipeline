import pytest
from src.llm.output_parser import OutputParser

def test_output_parser_strips_think():
    parser = OutputParser()
    raw = "<think>\nThinking about this claim...\n</think>\n```json\n{\n  \"root_cause\": \"test\",\n  \"carc_interpretation\": \"test\",\n  \"recoverability\": \"recoverable\",\n  \"confidence\": 0.9,\n  \"evidence\": [\"test\"],\n  \"recommended_action\": \"test\"\n}\n```"
    
    result = parser.parse_denial_analysis(raw, "CLM-1")
    assert result.recoverability == "recoverable"
    assert result.confidence == 0.9
    assert result.root_cause == "test"
    
def test_output_parser_fallback():
    parser = OutputParser()
    raw = "I'm sorry, I cannot fulfill this request."
    
    result = parser.parse_denial_analysis(raw, "CLM-1")
    assert result.recoverability == "needs_review"
    assert result.confidence == 0.0
    assert "LLM parsing failed" in result.root_cause
