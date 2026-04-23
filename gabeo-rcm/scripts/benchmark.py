import time
import json
import os
import psutil
import platform
import subprocess
import logging
from typing import Dict, List
from src.pipeline import Pipeline
from src.models import ClaimRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Benchmarker:
    def __init__(self):
        self.pipeline = Pipeline()
        self.results = {}

    def get_hardware_specs(self) -> Dict:
        """Detects CPU, RAM, and GPU specs."""
        specs = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu": "Not Detected"
        }
        
        # Try to detect NVIDIA GPU
        try:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"], encoding='utf-8')
            specs["gpu"] = gpu_info.strip()
        except:
            pass
            
        return specs

    def run_latency_trials(self, sample_claim_path: str, num_trials: int = 3) -> Dict:
        """Measures latency for individual pipeline components."""
        if not os.path.exists(sample_claim_path):
            logger.error(f"Sample claim not found at {sample_claim_path}")
            return {}

        try:
            with open(sample_claim_path, 'r', encoding='utf-8-sig') as f:
                claim_data = json.load(f)
        except UnicodeDecodeError:
            with open(sample_claim_path, 'r', encoding='utf-16') as f:
                claim_data = json.load(f)
        
        # We need a record
        record = self.pipeline.loader.parse_sample_json(sample_claim_path)
        
        latencies = {
            "p1_deterministic": [],
            "p1_llm_reasoning": [],
            "turbo_store_search": [],
            "p2_pattern_match": []
        }

        logger.info(f"Running {num_trials} latency trials...")

        for i in range(num_trials):
            # 1. TurboStore Search (P2 Matcher)
            start = time.perf_counter()
            self.pipeline.p2_matcher.get_similar_context(record)
            latencies["turbo_store_search"].append(time.perf_counter() - start)

            # 2. P1 Deterministic (Rule Engine)
            start = time.perf_counter()
            self.pipeline.p1_analyzer.run_rule_engine(record)
            latencies["p1_deterministic"].append(time.perf_counter() - start)

            # 3. P1 Full (LLM)
            start = time.perf_counter()
            self.pipeline.p1_analyzer.analyze_claim(record, similar_ctx="")
            latencies["p1_llm_reasoning"].append(time.perf_counter() - start)

        # Average results
        avg_results = {k: round(sum(v)/len(v), 4) for k, v in latencies.items() if v}
        return avg_results

    def run_throughput_benchmark(self, batch_path: str) -> Dict:
        """Measures wall-clock time and throughput for a batch."""
        if not os.path.exists(batch_path):
            logger.error(f"Batch file not found at {batch_path}")
            return {}

        try:
            with open(batch_path, 'r', encoding='utf-8-sig') as f:
                claims = json.load(f)
        except UnicodeDecodeError:
            with open(batch_path, 'r', encoding='utf-16') as f:
                claims = json.load(f)

        logger.info(f"Running throughput benchmark on {len(claims)} claims...")
        
        start_time = time.perf_counter()
        self.pipeline.run_batch_analysis(claims)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = len(claims) / total_time
        
        return {
            "batch_size": len(claims),
            "total_time_sec": round(total_time, 2),
            "claims_per_sec": round(throughput, 4),
            "sec_per_claim": round(total_time / len(claims), 2)
        }

    def run_full_benchmark(self, sample_claim: str, batch_file: str):
        """Runs the entire suite and saves results."""
        self.results["hardware"] = self.get_hardware_specs()
        self.results["latency_avg"] = self.run_latency_trials(sample_claim)
        self.results["throughput"] = self.run_throughput_benchmark(batch_file)
        
        report_path = "data/reports/performance_benchmark.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Benchmark Complete. Results saved to {report_path}")
        return self.results

if __name__ == "__main__":
    bench = Benchmarker()
    res = bench.run_full_benchmark("data/samples/CLM-2026-00142.json", "data/synthetic/claims.json")
    print(json.dumps(res, indent=2))
