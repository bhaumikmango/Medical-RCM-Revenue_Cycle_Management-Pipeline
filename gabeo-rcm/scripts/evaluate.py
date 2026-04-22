import json
import os
import logging
from typing import List, Dict
from src.pipeline import Pipeline
from src.models import DenialAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        self.pipeline = Pipeline()

    def run_evaluation(self, synthetic_data_path: str = "data/synthetic/claims.json"):
        """
        Runs the entire pipeline on synthetic data and calculates performance metrics.
        """
        if not os.path.exists(synthetic_data_path):
            logger.error(f"Synthetic data not found at {synthetic_data_path}")
            return

        with open(synthetic_data_path, 'r', encoding='utf-8') as f:
            claims_data = json.load(f)

        logger.info(f"Starting evaluation on {len(claims_data)} claims...")
        
        # We process them through the pipeline
        self.pipeline.run_batch_analysis(claims_data)
        
        # Now we pull the results from our claim store to evaluate
        results = self.pipeline.claim_store.get_all_denied_claims_with_analysis()
        
        metrics = {
            "total_processed": len(results),
            "recoverable": 0,
            "not_recoverable": 0,
            "needs_review": 0,
            "avg_confidence": 0.0,
            "carc_distribution": {}
        }

        conf_sum = 0
        for item in results:
            a = item['analysis']
            c = item['claim']
            
            metrics[a['recoverability']] += 1
            conf_sum += a['confidence']
            
            carc = c['carc_code']
            metrics['carc_distribution'][carc] = metrics['carc_distribution'].get(carc, 0) + 1

        metrics['avg_confidence'] = round(conf_sum / len(results), 2) if results else 0
        
        # Logic check: Verify if deterministic rules are being followed
        # (e.g., CARC 253 should always be not_recoverable)
        logic_errors = 0
        for item in results:
            a = item['analysis']
            c = item['claim']
            if c['carc_code'] == '253' and a['recoverability'] != 'not_recoverable':
                logic_errors += 1
            if c['carc_code'] == '252' and a['recoverability'] != 'recoverable':
                logic_errors += 1
        
        metrics['deterministic_rule_accuracy'] = round((len(results) - logic_errors) / len(results) * 100, 1) if results else 100

        logger.info("Evaluation Complete.")
        return metrics

if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.run_evaluation()
    print(json.dumps(metrics, indent=2))
