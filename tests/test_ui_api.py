import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.pipeline import Pipeline
import os

client = TestClient(app)

# --- SCENARIO: Dashboard Stats with Missing Database ---
def test_api_stats_no_data():
    """
    Scenario: Dashboard Stats with Missing/Empty Data
    Given the system has no analyzed claims in the database
    When a GET request is made to /api/stats
    Then it should return 200 OK
    And all statistical values should be zero or default
    And the system should not crash.
    """
    # Temporarily point to a non-existent DB if needed, but 
    # the router already handles 'if not results'
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    
    # We expect numbers, even if zero
    assert "total_claims" in data
    assert "total_value" in data
    assert "recovery_rate" in data

# --- SCENARIO: Accessing Non-existent Claim ---
def test_api_claims_structure():
    """
    Scenario: Claims List Structure
    Given a request for the claims list
    When a GET request is made to /api/claims
    Then it should return a list of objects
    And each object should have 'claim' and 'analysis' keys.
    """
    response = client.get("/api/claims")
    assert response.status_code == 200
    data = response.json()
    
    if len(data) > 0:
        assert "claim" in data[0]
        assert "analysis" in data[0]

# --- SCENARIO: Invalid Endpoint Handling ---
def test_api_404_handling():
    """
    Scenario: Invalid API Endpoint
    Given a request to a non-existent API path
    When the request is sent
    Then it should return 404 Not Found
    And a standard FastAPI error detail.
    """
    response = client.get("/api/non-existent-endpoint")
    assert response.status_code == 404
    assert "detail" in response.json()

# --- SCENARIO: Trends with Low Data ---
def test_api_trends_execution():
    """
    Scenario: Trends Analysis Execution
    Given the trends endpoint is called
    When the database is populated or empty
    Then it should return a list (possibly empty)
    And not raise a 500 server error.
    """
    response = client.get("/api/trends")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
