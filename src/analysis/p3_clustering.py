import os
import json
import logging
import re
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.models import ClaimRecord, DenialAnalysis
from src.llm.client import llm_client

logger = logging.getLogger(__name__)

class ClusterInsight:
    def __init__(self, cluster_id: int, num_claims: int, total_value: float,
                 dominant_payer: str, dominant_carc: str, dominant_procedure: str,
                 recovery_rate_pct: float, claims: List[str]):
        self.cluster_id = cluster_id
        self.num_claims = num_claims
        self.total_value = total_value
        self.dominant_payer = dominant_payer
        self.dominant_carc = dominant_carc
        self.dominant_procedure = dominant_procedure
        self.recovery_rate_pct = recovery_rate_pct
        self.claims = claims
        self.summary_json = {}

    def to_dict(self):
        d = self.__dict__.copy()
        return d

class DenialClusterer:
    CPT_LOOKUP = {
        "99213": "Office Visit - Level 3",
        "99214": "Office Visit - Level 4",
        "27447": "Total Knee Arthroplasty",
        "72148": "MRI Lumbar Spine",
        "99203": "New Patient Visit - Level 3",
        "99283": "ER Visit - Level 3",
        "G0439": "Annual Wellness Visit"
    }

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "p3_cluster_summary.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def _build_features(self, combined_data: List[Dict]) -> np.ndarray:
        """
        Builds the feature matrix for clustering.
        combined_data: List of dicts with 'claim' and 'analysis'
        """
        # Feature extraction
        import pandas as pd
        rows = []
        for d in combined_data:
            c = d['claim']
            a = d['analysis']
            
            # Map recoverability to a score
            rec_map = {"recoverable": 1.0, "needs_review": 0.5, "not_recoverable": 0.0}
            rec_score = rec_map.get(a.get('recoverability'), 0.0)
            
            rows.append({
                "carc_code": c.get('carc_code', 'Unknown'),
                "insurance_type": c.get('insurance_type', 'Unknown'),
                "adjustment_group": c.get('adjustment_group', 'Unknown'),
                "claim_amount": float(c.get('claim_amount', 0.0)),
                "recoverability_score": rec_score
            })
            
        df = pd.DataFrame(rows)
        
        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['claim_amount', 'recoverability_score']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 ['carc_code', 'insurance_type', 'adjustment_group'])
            ])
            
        features = preprocessor.fit_transform(df)
        return features

    def cluster_denials(self, combined_data: List[Dict]) -> List[ClusterInsight]:
        if not combined_data:
            return []
            
        n_clusters = min(self.n_clusters, len(combined_data))
        if n_clusters < 2:
            n_clusters = 1
            
        features = self._build_features(combined_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        clusters_data = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(labels):
            clusters_data[label].append(combined_data[i])
            
        insights = []
        for cluster_id, items in clusters_data.items():
            if not items:
                continue
                
            num_claims = len(items)
            total_value = sum(float(item['claim'].get('claim_amount', 0.0)) for item in items)
            
            payers = [item['claim'].get('payer_name', '') for item in items]
            dominant_payer = max(set(payers), key=payers.count) if payers else "Unknown"
            
            carcs = [item['claim'].get('carc_code', '') for item in items]
            dominant_carc = max(set(carcs), key=carcs.count) if carcs else "Unknown"
            
            procs = [item['claim'].get('procedure_code', '') for item in items]
            dominant_procedure = max(set(procs), key=procs.count) if procs else "Unknown"
            procedure_desc = self.CPT_LOOKUP.get(dominant_procedure, "Unknown Procedure")
            
            rec_scores = [{"recoverable": 1.0, "needs_review": 0.5, "not_recoverable": 0.0}.get(item['analysis'].get('recoverability'), 0.0) for item in items]
            recovery_rate_pct = int((sum(rec_scores) / len(rec_scores)) * 100) if rec_scores else 0
            
            claim_ids = [item['claim'].get('claim_id') for item in items]
            
            # Sample 3 representative claims
            import random
            sample_size = min(3, len(items))
            samples = random.sample(items, sample_size)
            rep_claims_str = ""
            for s in samples:
                rep_claims_str += f"- {s['claim']['claim_id']}: {s['claim']['payer_name']}, CPT {s['claim']['procedure_code']}, CARC {s['claim']['carc_code']}, Verdict: {s['analysis']['recoverability']}\n"

            insight = ClusterInsight(
                cluster_id=cluster_id,
                num_claims=num_claims,
                total_value=round(total_value, 2),
                dominant_payer=dominant_payer,
                dominant_carc=dominant_carc,
                dominant_procedure=dominant_procedure,
                recovery_rate_pct=recovery_rate_pct,
                claims=claim_ids
            )
            
            # Ask LLM to summarize
            prompt = self.prompt_template.format(
                num_claims=num_claims,
                total_value=round(total_value, 2),
                dominant_payer=dominant_payer,
                dominant_carc=dominant_carc,
                dominant_procedure=dominant_procedure,
                procedure_desc=procedure_desc,
                recovery_rate_pct=recovery_rate_pct,
                representative_claims=rep_claims_str or "None available."
            )
            
            raw_output = llm_client.generate_sync(prompt)
            insight.summary_json = self._parse_llm_json(raw_output)
            
            insights.append(insight)
            
        return insights

    def _parse_llm_json(self, raw: str) -> Dict:
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except:
            return {
                "cluster_label": "Failed to parse",
                "dominant_denial_reason": "Failed to parse LLM response",
                "recovery_opportunity": "Error",
                "estimated_recovery_rate": 0,
                "recommended_bulk_action": "Error"
            }
