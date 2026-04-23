from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ClaimRecord:
    # 835 fields (pc_/pcl_/pcla_/cp_ prefixes)
    claim_id: str
    payer_name: str
    claim_status: str
    claim_amount: float
    claim_paid: float
    insurance_type: str
    received_date: str
    statement_begin: str
    carc_code: str
    carc_description: str
    adjustment_group: str
    adjustment_amount: float
    procedure_code: str
    procedure_modifier: str
    remark_codes: str
    
    # 837 fields (ec_ prefix)
    service_date_from: str
    principal_diagnosis: str
    diag2: str
    prior_authorization: str
    delay_reason_code: str
    claim_frequency: str
    bill_prov_npi: str
    subscriber_id: str
    type_of_bill: str
    rend_prov_specialty: str
    original_ref: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DenialAnalysis:
    claim_id: str
    root_cause: str
    carc_interpretation: str
    recoverability: str
    confidence: float
    evidence: List[str]
    similar_claims: List[Dict] = field(default_factory=list)
    recommended_action: str = ""
    rule_engine_flags: Dict = field(default_factory=dict)
