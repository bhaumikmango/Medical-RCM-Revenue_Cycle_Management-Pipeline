import csv
import os

class CarcLookup:
    def __init__(self, csv_path: str):
        self.lookup = {}
        self.load(csv_path)

    def load(self, csv_path: str):
        if not os.path.exists(csv_path):
            # Fallback to hardcoded values for the core 16 codes if file missing
            self._load_fallback()
            return
            
        # Parse the raw text file (CARC Codes to be used.txt)
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('* **'):
                # Format: * **16**: Claim/service lacks information...
                parts = line.split('**: ')
                if len(parts) == 2:
                    code = parts[0].replace('* **', '')
                    desc = parts[1]
                    self.lookup[code] = {
                        "description": desc,
                        "category": "Unknown",
                        "status": "deactivated" if code == "15" else "active"
                    }

    def _load_fallback(self):
        # The 16 required codes
        self.lookup = {
            "4": {"description": "The procedure code is inconsistent with the modifier used.", "status": "active"},
            "11": {"description": "The diagnosis is inconsistent with the procedure.", "status": "active"},
            "16": {"description": "Claim/service lacks information or has submission/billing error(s).", "status": "active"},
            "18": {"description": "Exact duplicate claim/service", "status": "active"},
            "19": {"description": "This is a work-related injury/illness and thus the liability of the Worker's Compensation Carrier.", "status": "active"},
            "22": {"description": "This care may be covered by another payer per coordination of benefits.", "status": "active"},
            "27": {"description": "Expenses were incurred after coverage was terminated.", "status": "active"},
            "29": {"description": "The time limit for filing has expired.", "status": "active"},
            "45": {"description": "Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement.", "status": "active"},
            "50": {"description": "These are non-covered services because this is not deemed a 'medical necessity' by the payer.", "status": "active"},
            "96": {"description": "Non-covered charge(s).", "status": "active"},
            "97": {"description": "The benefit for this service is included in the payment/allowance for another service/procedure that has already been adjudicated.", "status": "active"},
            "150": {"description": "Payer deems the information submitted does not support this level of service.", "status": "active"},
            "197": {"description": "Precertification/authorization/notification absent.", "status": "active"},
            "252": {"description": "An attachment/other documentation is required to adjudicate this claim/service.", "status": "active"},
            "253": {"description": "Sequestration - reduction in federal payment.", "status": "active"}
        }

    def get(self, code: str) -> dict:
        code = str(code).strip()
        return self.lookup.get(code, {"description": "Unknown CARC Code", "status": "unknown"})
