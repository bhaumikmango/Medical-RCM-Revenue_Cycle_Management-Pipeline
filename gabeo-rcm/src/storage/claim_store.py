import sqlite3
import json
import os
from typing import List, Dict
from src.models import ClaimRecord, DenialAnalysis

class ClaimStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "claims.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    payer_name TEXT,
                    insurance_type TEXT,
                    claim_amount REAL,
                    carc_code TEXT,
                    procedure_code TEXT,
                    raw_data TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    claim_id TEXT PRIMARY KEY,
                    root_cause TEXT,
                    recoverability TEXT,
                    confidence REAL,
                    recommended_action TEXT,
                    raw_data TEXT,
                    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
                )
            """)
            conn.commit()

    def save_claim(self, claim: ClaimRecord):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO claims 
                (claim_id, payer_name, insurance_type, claim_amount, carc_code, procedure_code, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.claim_id, claim.payer_name, claim.insurance_type, 
                claim.claim_amount, claim.carc_code, claim.procedure_code,
                json.dumps(claim.to_dict())
            ))
            conn.commit()

    def save_analysis(self, analysis: DenialAnalysis):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO analyses 
                (claim_id, root_cause, recoverability, confidence, recommended_action, raw_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis.claim_id, analysis.root_cause, analysis.recoverability,
                analysis.confidence, analysis.recommended_action,
                json.dumps(analysis.__dict__)
            ))
            conn.commit()

    def get_all_denied_claims_with_analysis(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.raw_data as claim_json, a.raw_data as analysis_json
                FROM claims c
                JOIN analyses a ON c.claim_id = a.claim_id
            """)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "claim": json.loads(row["claim_json"]),
                    "analysis": json.loads(row["analysis_json"])
                })
            return results
