import argparse
import json
import os
import sys
import logging
from src.pipeline import Pipeline
from scripts.evaluate import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RCM-CLI")

def main():
    parser = argparse.ArgumentParser(description="Gabeo AI — Al-Powered Claim Denial Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze single claim
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single claim")
    analyze_parser.add_argument("--claim", required=True, help="Path to JSON file containing single claim (835 + 837 blocks)")

    # Batch analysis
    batch_parser = subparsers.add_parser("batch", help="Run batch analysis on a set of claims")
    batch_parser.add_argument("--input", required=True, help="Path to JSON file containing list of claims")
    batch_parser.add_argument("--cluster", action="store_true", help="Run P3 clustering after analysis")

    # Trends
    trends_parser = subparsers.add_parser("trends", help="Generate systemic denial trend report (P2.3)")
    trends_parser.add_argument("--min-claims", type=int, default=2, help="Minimum claim volume to report a trend")
    trends_parser.add_argument("--output", default="data/reports/trends_report.json", help="Output path for the report")

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Run pipeline evaluation against synthetic data")

    # UI
    ui_parser = subparsers.add_parser("ui", help="Launch the RCM Analysis Dashboard (Web UI)")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    pipeline = Pipeline()

    if args.command == "analyze":
        if not os.path.exists(args.claim):
            logger.error(f"File not found: {args.claim}")
            return
        with open(args.claim, 'r', encoding='utf-8') as f:
            claim_data = json.load(f)
        
        # Format expects a list for batch analysis or we can add a specific method
        # For simplicity, we wrap in list and run
        pipeline.run_batch_analysis([claim_data])
        
        # Get the result from store
        results = pipeline.claim_store.get_all_denied_claims_with_analysis()
        # Find the one we just did (last one)
        last_result = results[-1]
        print(json.dumps(last_result["analysis"], indent=2))

    elif args.command == "batch":
        if not os.path.exists(args.input):
            logger.error(f"File not found: {args.input}")
            return
        with open(args.input, 'r', encoding='utf-8') as f:
            claims_list = json.load(f)
        
        pipeline.run_batch_analysis(claims_list)
        logger.info(f"Batch analysis complete for {len(claims_list)} claims.")

        if args.cluster:
            logger.info("Running P3 Clustering...")
            clusters = pipeline.run_clustering()
            for c in clusters:
                print(f"\n--- Cluster: {c.summary_json.get('cluster_label')} ---")
                print(f"Impact: {c.num_claims} claims | ${c.total_value}")
                print(f"Opportunity: {c.summary_json.get('recovery_opportunity')}")
                print(f"Action: {c.summary_json.get('recommended_bulk_action')}")

    elif args.command == "trends":
        logger.info("Generating Systemic Trend Report...")
        trends = pipeline.run_trend_analysis(min_claims=args.min_claims)
        pipeline.trend_reporter.save_report(trends, output_path=args.output)
        print(f"Generated {len(trends)} systemic patterns. Report saved to {args.output}")

    elif args.command == "evaluate":
        evaluator = Evaluator()
        metrics = evaluator.run_evaluation()
        print("\n=== Pipeline Performance Metrics ===")
        print(json.dumps(metrics, indent=2))

    elif args.command == "ui":
        import uvicorn
        logger.info(f"Launching Dashboard on http://localhost:{args.port}")
        # Updated to point to the new app directory structure
        uvicorn.run("app.main:app", host="0.0.0.0", port=args.port, reload=True)

if __name__ == "__main__":
    main()
