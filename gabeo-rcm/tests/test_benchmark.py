import pytest
from unittest.mock import MagicMock, patch
from scripts.benchmark import Benchmarker

def test_benchmark_hardware_detection():
    bench = Benchmarker()
    specs = bench.get_hardware_specs()
    
    assert "os" in specs
    assert "cpu_cores" in specs
    assert specs["cpu_cores"] > 0
    assert "ram_total_gb" in specs

def test_benchmark_latency_math():
    bench = Benchmarker()
    
    # Mock pipeline and its components
    bench.pipeline = MagicMock()
    
    # Mock sample loading
    with patch("builtins.open", MagicMock()):
        with patch("json.load", return_value={"835": {}, "837": {}}):
            with patch("os.path.exists", return_value=True):
                # We need to mock parse_sample_json to return something
                bench.pipeline.loader.parse_sample_json.return_value = MagicMock()
                
                # Run trials (1 trial for speed)
                results = bench.run_latency_trials("fake.json", num_trials=1)
                
                assert "p1_deterministic" in results
                assert "p1_llm_reasoning" in results
                assert "turbo_store_search" in results

def test_throughput_calculation():
    bench = Benchmarker()
    bench.pipeline = MagicMock()
    
    with patch("builtins.open", MagicMock()):
        with patch("json.load", return_value=[{}, {}, {}]): # 3 claims
            with patch("os.path.exists", return_value=True):
                # Speed up by mocking time.perf_counter
                with patch("time.perf_counter", side_effect=[0, 1]): # 1 second total
                    res = bench.run_throughput_benchmark("fake_batch.json")
                    
                    assert res["batch_size"] == 3
                    assert res["total_time_sec"] == 1.0
                    assert res["claims_per_sec"] == 3.0
