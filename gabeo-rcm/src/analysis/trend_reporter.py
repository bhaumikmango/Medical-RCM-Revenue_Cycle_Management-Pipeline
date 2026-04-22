import sqlite3
import pandas as pd
import os
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TrendReporter:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "claims.db")
        self.db_path = db_path

    def generate_systemic_trends(self, min_claims: int = 2) -> List[Dict]:
        """
        Queries the claim store and identifies systemic denial patterns.
        Groups by Payer, Procedure, and CARC.
        """
        if not os.path.exists(self.db_path):
            logger.warning("Claim database not found. Cannot generate trends.")
            return []

        with sqlite3.connect(self.db_path) as conn:
            # We join claims and analyses to get a full picture
            query = """
                SELECT 
                    c.payer_name, 
                    c.procedure_code, 
                    c.carc_code,
                    c.claim_amount,
                    a.recoverability
                FROM claims c
                JOIN analyses a ON c.claim_id = a.claim_id
            """
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return []

        # Grouping to find patterns
        # We define a "pattern" as a Payer + Procedure + CARC combination
        summary = df.groupby(['payer_name', 'procedure_code', 'carc_code']).agg(
            total_claims=('claim_amount', 'count'),
            avg_denied_amount=('claim_amount', 'mean'),
            recoverable_count=('recoverability', lambda x: (x == 'recoverable').sum())
        ).reset_index()

        # Filter by minimum volume to ensure statistical relevance
        summary = summary[summary['total_claims'] >= min_claims]

        # Calculate denial rate (all claims in this store are denied claims by definition, 
        # so "denial_rate" here refers to the systemic nature of this specific reason)
        # However, to match SRS language, we'll present it as the "Pattern Frequency"
        
        results = []
        for _, row in summary.iterrows():
            results.append({
                "payer": row['payer_name'],
                "procedure_code": row['procedure_code'],
                "carc_code": row['carc_code'],
                "total_claims": int(row['total_claims']),
                "avg_denied_amount": round(float(row['avg_denied_amount']), 2),
                "systemic_frequency_score": round(int(row['total_claims']) / len(df) * 100, 1), # % of total denials in this batch
                "historical_recovery_rate": round(int(row['recoverable_count']) / int(row['total_claims']) * 100, 1)
            })

        # Sort by total impact (total claims * avg amount)
        results.sort(key=lambda x: x['total_claims'] * x['avg_denied_amount'], reverse=True)
        
        return results

    def save_report(self, results: List[Dict], output_path: str = "data/reports/trends_report.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Trend report saved to {output_path}")
