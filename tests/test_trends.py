import pytest
import os
import sqlite3
import pandas as pd
from src.analysis.trend_reporter import TrendReporter

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_trends.db"
    # Create mock data
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE claims (claim_id TEXT, payer_name TEXT, procedure_code TEXT, carc_code TEXT, claim_amount REAL)")
    cursor.execute("CREATE TABLE analyses (claim_id TEXT, recoverability TEXT)")
    
    # 3 claims of same pattern (Aetna, 99213, 29)
    for i in range(3):
        cursor.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?)", (f"C{i}", "Aetna", "99213", "29", 100.0))
        cursor.execute("INSERT INTO analyses VALUES (?, ?)", (f"C{i}", "recoverable" if i < 2 else "needs_review"))
        
    # 1 claim of different pattern
    cursor.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?)", ("C3", "UHC", "99214", "16", 500.0))
    cursor.execute("INSERT INTO analyses VALUES (?, ?)", ("C3", "not_recoverable"))
    
    conn.commit()
    conn.close()
    return str(db_file)

def test_trend_reporting_logic(temp_db):
    reporter = TrendReporter(db_path=temp_db)
    trends = reporter.generate_systemic_trends(min_claims=2)
    
    assert len(trends) == 1
    t = trends[0]
    assert t["payer"] == "Aetna"
    assert t["procedure_code"] == "99213"
    assert t["total_claims"] == 3
    assert t["historical_recovery_rate"] == 66.7 # 2 out of 3 recoverable
    assert t["avg_denied_amount"] == 100.0

def test_trend_reporting_min_claims(temp_db):
    reporter = TrendReporter(db_path=temp_db)
    # With min_claims=1, we should see both patterns
    trends = reporter.generate_systemic_trends(min_claims=1)
    assert len(trends) == 2
    payers = [tr["payer"] for tr in trends]
    assert "Aetna" in payers
    assert "UHC" in payers
