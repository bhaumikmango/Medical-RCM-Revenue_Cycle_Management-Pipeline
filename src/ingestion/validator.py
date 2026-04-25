import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class Validator:
    REQUIRED_FIELDS = [
        'claim_id', 'carc_code', 'claim_amount', 
        'received_date', 'service_date_from', 'insurance_type'
    ]

    def validate(self, claim_dict: Dict) -> Dict:
        """
        Validates a claim record dictionary.
        Returns a dictionary with 'is_valid' boolean and 'warnings'/'errors' lists.
        """
        result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }

        # 1. Required fields check
        for field in self.REQUIRED_FIELDS:
            if not claim_dict.get(field):
                result["errors"].append(f"Missing required field: {field}")
                result["is_valid"] = False

        # 2. Special Rules
        carc = str(claim_dict.get("carc_code", ""))
        remark = str(claim_dict.get("remark_codes", "")).strip()
        
        # CARC 16 and 96 MUST have remark codes
        if carc in ("16", "96") and not remark:
            result["warnings"].append("MISSING_REMARK_CODE")
            
        # Legacy/Deactivated codes
        if carc == "15":
            result["warnings"].append("DEACTIVATED_CARC_15")

        return result
