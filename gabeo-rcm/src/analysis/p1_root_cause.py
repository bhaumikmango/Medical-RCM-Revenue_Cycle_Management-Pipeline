import json
import os
import datetime
from src.models import ClaimRecord, DenialAnalysis
from src.llm.client import llm_client
from src.llm.output_parser import OutputParser
from src.ingestion.carc_lookup import CarcLookup

class RootCauseAnalyzer:
    def __init__(self, carc_lookup: CarcLookup):
        self.carc_lookup = carc_lookup
        self.parser = OutputParser()
        
        # Load prompt template
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "p1_root_cause.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def _parse_date(self, date_str: str) -> datetime.date:
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            return None

    def run_rule_engine(self, claim: ClaimRecord) -> dict:
        """
        Runs deterministic logic to pre-fill factual conclusions before the LLM step.
        """
        flags = {"confidence": None, "recoverability": None, "evidence": []}
        carc = claim.carc_code
        
        # CARC 29: Timely Filing
        if carc == "29":
            received = self._parse_date(claim.received_date)
            service = self._parse_date(claim.service_date_from)
            if received and service:
                days = (received - service).days
                flags["evidence"].append(f"Filing gap: {days} days")
                flags["confidence"] = 1.0
                
                is_medicare = "medicare" in claim.insurance_type.lower()
                limit = 365 if is_medicare else 180
                
                if days > limit:
                    flags["recoverability"] = "not_recoverable"
                    flags["evidence"].append(f"Exceeded {limit} day limit for {claim.insurance_type}")
                else:
                    flags["recoverability"] = "recoverable"
                    flags["evidence"].append(f"Within {limit} day limit for {claim.insurance_type}")
            else:
                flags["recoverability"] = "needs_review"
                flags["evidence"].append("Invalid dates for timely filing calculation")
                
        # CARC 18: Duplicate
        elif carc == "18":
            # No original reference fields provided in our simplified ClaimRecord, 
            # so we always default to needs_review as per decision Q1
            flags["recoverability"] = "needs_review"
            flags["confidence"] = 0.7
            flags["evidence"].append("No original claim reference found. Needs manual verification of duplicate status.")
            
        # CARC 16: Missing Info
        elif carc == "16":
            missing = []
            if not claim.principal_diagnosis: missing.append("Principal Diagnosis")
            if not claim.bill_prov_npi: missing.append("Billing Provider NPI")
            if not claim.subscriber_id: missing.append("Subscriber ID")
            if not claim.procedure_modifier: missing.append("Procedure Modifier")
            
            if missing:
                flags["evidence"].extend([f"Missing field: {m}" for m in missing])
            flags["confidence"] = 0.9
            flags["recoverability"] = "needs_review"
            
        # CARC 197: Auth Absent
        elif carc == "197":
            if not claim.prior_authorization:
                flags["evidence"].append("Prior authorization field is completely empty.")
                flags["confidence"] = 0.9
                flags["recoverability"] = "not_recoverable"
                
        # CARC 50: Medical Necessity
        elif carc == "50":
            if not claim.prior_authorization:
                flags["evidence"].append("No prior authorization found for potential medical necessity.")
                flags["confidence"] = 0.85
                flags["recoverability"] = "needs_review"
                
        # CARC 96: Non-covered
        elif carc == "96":
            if not claim.remark_codes:
                flags["evidence"].append("MISSING_REMARK_CODE: CARC 96 requires a remark code per EDI standards.")
            flags["confidence"] = 0.9
            flags["recoverability"] = "not_recoverable"
            
        # CARC 4: Modifier
        elif carc == "4":
            if not claim.procedure_modifier:
                flags["evidence"].append("Procedure modifier field is empty.")
                flags["confidence"] = 0.8
                flags["recoverability"] = "needs_review"
                
        # CARC 97: Bundled
        elif carc == "97":
            flags["confidence"] = 0.75
            flags["recoverability"] = "needs_review"
            
        # CARC 252: Attachment
        elif carc == "252":
            flags["confidence"] = 0.9
            flags["recoverability"] = "recoverable"
            flags["evidence"].append("Documentation requested. Can be recovered by submitting attachment.")
            
        # CARC 253: Sequestration
        elif carc == "253":
            flags["confidence"] = 1.0
            flags["recoverability"] = "not_recoverable"
            flags["evidence"].append("Federal budget sequestration reduction. Not appealable.")

        return flags

    def build_prompt(self, claim: ClaimRecord, rule_flags: dict, similar_ctx: str) -> str:
        carc_info = self.carc_lookup.get(claim.carc_code)
        
        return self.prompt_template.format(
            claim_json=json.dumps(claim.to_dict(), indent=2),
            carc_code=claim.carc_code,
            carc_description=carc_info["description"],
            adjustment_group=claim.adjustment_group,
            rule_engine_flags=json.dumps(rule_flags, indent=2),
            similar_claims_context=similar_ctx or "None provided."
        )

    def analyze_claim(self, claim: ClaimRecord, similar_ctx: str = "") -> DenialAnalysis:
        rule_flags = self.run_rule_engine(claim)
        prompt = self.build_prompt(claim, rule_flags, similar_ctx)
        
        raw_output = llm_client.generate_sync(prompt)
        
        # Parse and override LLM if rule engine has deterministic fields
        analysis = self.parser.parse_denial_analysis(raw_output, claim.claim_id, rule_flags)
        
        # Hard override for CARC 252 as per plan
        if claim.carc_code == "252":
            analysis.recoverability = "recoverable"
            analysis.confidence = 0.9
            
        # If rule engine has a deterministic verdict and high confidence, prefer it
        if rule_flags["recoverability"] and rule_flags["confidence"] and rule_flags["confidence"] >= 0.9:
            analysis.recoverability = rule_flags["recoverability"]
            # Blend evidence
            for ev in rule_flags["evidence"]:
                if ev not in analysis.evidence:
                    analysis.evidence.insert(0, ev)
                    
        return analysis
