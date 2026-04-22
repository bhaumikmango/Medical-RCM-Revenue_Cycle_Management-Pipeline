import json
import re
import logging
from typing import Dict, Any
from src.models import DenialAnalysis

logger = logging.getLogger(__name__)

class OutputParser:
    REQUIRED_KEYS = ["root_cause", "carc_interpretation", "recoverability", "confidence", "evidence", "recommended_action"]

    def parse_denial_analysis(self, raw_output: str, claim_id: str, rule_flags: dict = None) -> DenialAnalysis:
        """
        Cleans the raw LLM output and validates it against the schema.
        Handles deepseek-r1's <think> tags.
        """
        cleaned = raw_output

        # Step 1: Strip deepseek-r1 <think>...</think> tags if present
        # Use DOTALL to match across newlines
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        
        # Step 2: Strip markdown JSON fences if present
        cleaned = cleaned.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        cleaned = cleaned.strip()

        # Step 3: Parse and Validate
        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {claim_id}. Raw output: {raw_output}")
            
        if parsed and all(k in parsed for k in self.REQUIRED_KEYS):
            # Valid parse
            return DenialAnalysis(
                claim_id=claim_id,
                root_cause=str(parsed["root_cause"]),
                carc_interpretation=str(parsed["carc_interpretation"]),
                recoverability=str(parsed["recoverability"]),
                confidence=float(parsed["confidence"]),
                evidence=[str(e) for e in parsed["evidence"]],
                recommended_action=str(parsed["recommended_action"]),
                rule_engine_flags=rule_flags or {}
            )
            
        # Fallback if invalid format or parse failure
        logger.warning(f"Using fallback parser for {claim_id}. Output was invalid.")
        return DenialAnalysis(
            claim_id=claim_id,
            root_cause="LLM parsing failed.",
            carc_interpretation="N/A",
            recoverability="needs_review",
            confidence=0.0,
            evidence=["Parsing error: The model did not output valid JSON matching the schema."],
            recommended_action="Manual review required.",
            rule_engine_flags=rule_flags or {}
        )
