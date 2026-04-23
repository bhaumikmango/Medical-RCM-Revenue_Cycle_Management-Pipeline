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

    def run_evaluation(self, synthetic_data_path: str = "data/synthetic/claims.json", ground_truth_path: str = "data/synthetic/ground_truth.json"):
        """
        Runs the entire pipeline on synthetic data and calculates performance metrics.
        """
        if not os.path.exists(synthetic_data_path):
            logger.error(f"Synthetic data not found at {synthetic_data_path}")
            return
        
        if not os.path.exists(ground_truth_path):
            logger.error(f"Ground truth not found at {ground_truth_path}")
            return

        with open(synthetic_data_path, 'r', encoding='utf-8') as f:
            claims_data = json.load(f)
            
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        logger.info(f"Starting evaluation on {len(claims_data)} claims...")
        
        # Process through pipeline
        self.pipeline.run_batch_analysis(claims_data)
        
        # Pull results from store
        results = self.pipeline.claim_store.get_all_denied_claims_with_analysis()
        
        y_true = []
        y_pred = []
        confidences = []
        
        metrics = {
            "total_processed": len(results),
            "avg_confidence": 0.0,
            "deterministic_rule_accuracy": 0.0,
            "classification_report": {},
            "confidence_calibration": {}
        }

        conf_sum = 0
        logic_errors = 0
        
        for item in results:
            a = item['analysis']
            c = item['claim']
            claim_id = c['claim_id']
            
            if claim_id in ground_truth:
                y_true.append(ground_truth[claim_id]['recoverability'])
                y_pred.append(a['recoverability'])
                confidences.append(a['confidence'])
                
            conf_sum += a['confidence']
            
            # Deterministic checks
            if c['carc_code'] == '253' and a['recoverability'] != 'not_recoverable':
                logic_errors += 1
            if c['carc_code'] == '252' and a['recoverability'] != 'recoverable':
                logic_errors += 1

        metrics['avg_confidence'] = round(conf_sum / len(results), 2) if results else 0
        metrics['deterministic_rule_accuracy'] = round((len(results) - logic_errors) / len(results) * 100, 1) if results else 100
        
        # Tracking specific failures
        mismatches = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                claim_id = list(ground_truth.keys())[i] # Approximate, better to track in loop
                mismatches.append({
                    "claim_id": claim_id,
                    "actual": y_true[i],
                    "predicted": y_pred[i],
                    "confidence": confidences[i]
                })
        
        metrics['failures'] = mismatches
        
        # LLM Quality Metrics (Classification Report)
        if y_true and y_pred:
            from sklearn.metrics import classification_report
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # Confidence Calibration (Banding)
            bands = [
                ("0.0-0.5", 0.0, 0.5),
                ("0.5-0.7", 0.5, 0.7),
                ("0.7-0.9", 0.7, 0.9),
                ("0.9-1.0", 0.9, 1.1)
            ]
            
            for name, low, high in bands:
                band_correct = 0
                band_total = 0
                for i in range(len(confidences)):
                    if low <= confidences[i] < high:
                        band_total += 1
                        if y_true[i] == y_pred[i]:
                            band_correct += 1
                
                metrics['confidence_calibration'][name] = {
                    "count": band_total,
                    "accuracy": round(band_correct / band_total, 2) if band_total > 0 else 0.0
                }

        # Save to reports
        report_path = "data/reports/evaluation_results.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation Complete. Detailed results saved to {report_path}")
        return metrics

if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.run_evaluation()
    print(json.dumps(metrics, indent=2))
