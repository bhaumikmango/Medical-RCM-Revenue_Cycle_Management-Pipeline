import json
import pandas as pd
from typing import Dict, List, Optional
from src.models import ClaimRecord
from src.ingestion.validator import Validator

class Loader:
    def __init__(self):
        self.validator = Validator()

    def parse_sample_json(self, json_path: str) -> Optional[ClaimRecord]:
        """
        Parses the specific sample format with '835' and '837' blocks.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return self._to_claim_record(data.get("835", {}), data.get("837", {}))
        except Exception as e:
            print(f"Error parsing sample JSON {json_path}: {e}")
            return None

    def _safe_float(self, val, default=0.0):
        try:
            if pd.isna(val): return default
            return float(val)
        except:
            return default
            
    def _safe_str(self, val):
        if pd.isna(val): return ""
        return str(val).strip()

    def _to_claim_record(self, d835: Dict, d837: Dict) -> ClaimRecord:
        """
        Maps dictionary fields to the ClaimRecord dataclass based on the exact
        prefixes and fields defined in the prompt.
        """
        return ClaimRecord(
            claim_id=self._safe_str(d835.get("pc_ClaimID") or d837.get("ec_ClaimNo")),
            payer_name=self._safe_str(d837.get("ec_PayerName") or d835.get("cp_PayerName")),
            claim_status=self._safe_str(d835.get("pc_ClaimStatus")),
            claim_amount=self._safe_float(d835.get("pc_ClaimAmount") or d835.get("pcl_ChargedAmount") or d837.get("ec_Amount")),
            claim_paid=self._safe_float(d835.get("pc_ClaimPaid") or d835.get("pcl_PaidAmount")),
            insurance_type=self._safe_str(d835.get("pc_InsuranceType") or d837.get("ec_InsuranceType")),
            received_date=self._safe_str(d835.get("pc_ReceivedDate")),
            statement_begin=self._safe_str(d835.get("pc_StatementBegin")),
            carc_code=self._safe_str(d835.get("pcla_AdjustmentReason")),
            carc_description="", # Filled by carc lookup later
            adjustment_group=self._safe_str(d835.get("pcla_AdjustmentGroup")),
            adjustment_amount=self._safe_float(d835.get("pcla_AdjustmentAmount")),
            procedure_code=self._safe_str(d835.get("pcl_ProcedureCode") or d837.get("cd_ProcedureCode")),
            procedure_modifier=self._safe_str(d835.get("pcl_ProcedureModifier1") or d837.get("cd_Modifier1")),
            remark_codes=self._safe_str(d835.get("pcl_RemarkCodes")),
            service_date_from=self._safe_str(d837.get("ec_ServiceDateFrom")),
            principal_diagnosis=self._safe_str(d837.get("ec_PrincipalDiagnosis")),
            diag2=self._safe_str(d837.get("ec_Diag2")),
            prior_authorization=self._safe_str(d837.get("ec_PriorAuthorization") or d835.get("pc_PriorAuthNum")),
            delay_reason_code=self._safe_str(d837.get("ec_DelayReasonCode")),
            claim_frequency=self._safe_str(d837.get("ec_ClaimFrequency")),
            bill_prov_npi=self._safe_str(d837.get("ec_BillProvNPI")),
            subscriber_id=self._safe_str(d837.get("ec_SubscriberID")),
            type_of_bill=self._safe_str(d837.get("ec_TypeOfBill")),
            rend_prov_specialty=self._safe_str(d837.get("ec_RendProvSpecialty")),
            original_ref=self._safe_str(d835.get("pc_OrigRefNo"))
        )

    def join_claims_from_csv(self, df_835: pd.DataFrame, df_837: pd.DataFrame) -> List[ClaimRecord]:
        """
        Joins full pandas dataframes on the claim ID and returns instantiated ClaimRecords.
        """
        # pc_ClaimID is col 121 in 835. ec_ClaimNo is col 720 in 837.
        merged = pd.merge(
            df_835, df_837,
            left_on="pc_ClaimID",
            right_on="ec_ClaimNo",
            how="left"
        )
        
        records = []
        for _, row in merged.iterrows():
            row_dict = row.to_dict()
            records.append(self._to_claim_record(row_dict, row_dict))
            
        return records
