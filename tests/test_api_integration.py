# --- File: tests/test_api_integration.py (Placeholder) ---
import pytest
import requests
import time
import json

# Define the base URL for your API service
API_BASE_URL = "http://localhost:8081"  # This should match the host port exposed in docker-compose.yml


@pytest.fixture(scope="module")
def wait_for_api():
    """Wait until the API service is healthy."""
    url = f"{API_BASE_URL}/health"
    max_retries = 30
    retry_delay_sec = 2
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200 and response.json().get("status") == "healthy":
                print(f"\nAPI is healthy after {i * retry_delay_sec} seconds.")
                return
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting for API to be healthy... (attempt {i + 1}/{max_retries})")
        time.sleep(retry_delay_sec)
    pytest.fail(f"API did not become healthy within {max_retries * retry_delay_sec} seconds.")


def test_api_health_endpoint(wait_for_api):
    """Test the /health endpoint of the API."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["decision_engine_loaded"] is True
    # LLM client might be loaded or not depending on .env setup, so check existence
    assert "llm_client_loaded" in data
    print(f"\n✅ API Health Check passed. Details: {data}")


def test_recommend_endpoint_success(wait_for_api):
    """Test the /recommend endpoint with valid inputs."""
    url = f"{API_BASE_URL}/recommend"
    payload = {
        "scenario_name": "Small_Office",
        "annual_demand_kwh": 250000,
        "user_grid_dependency_pct": 30,
        "esg_weights": {"environment": 0.25, "social": 0.25, "governance": 0.25, "cost": 0.25}
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation"] is not None
    assert "total_cost" in data["recommendation"]
    assert "self_sufficiency_pct" in data["recommendation"]
    print(f"\n✅ /recommend endpoint returned a valid recommendation for Small_Office.")


def test_recommend_endpoint_no_solution(wait_for_api):
    """Test /recommend endpoint when no feasible solution should be found."""
    url = f"{API_BASE_URL}/recommend"
    payload = {
        "scenario_name": "Small_Office",
        "annual_demand_kwh": 250000,
        "user_grid_dependency_pct": 0,  # Very strict self-sufficiency
        "esg_weights": {"environment": 1.0, "social": 0.0, "governance": 0.0, "cost": 0.0}
    }
    response = requests.post(url, json=payload)
    # The API is designed to return 404 if no solution found
    assert response.status_code == 404
    data = response.json()
    assert data["recommendation"] is None
    assert "No solutions meet the" in data["status"] or "No solutions found with a reasonable payback period" in data[
        "status"]
    print(f"\n✅ /recommend endpoint correctly handled 'no solution found' case.")


def test_chat_endpoint_success(wait_for_api):
    """Test the /chat endpoint with a natural language query."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        "query": "Find a low cost system for a university campus with 3 million kWh annual demand."
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"]["summary"] is not None
    assert isinstance(data["response"]["details"], list)
    print(f"\n✅ /chat endpoint returned a valid AI response.")


def test_ml_predict_endpoint_success(wait_for_api):
    """Test the /predict_ml endpoint with valid inputs."""
    url = f"{API_BASE_URL}/predict_ml"
    payload = {
        "scenario_name": "Hospital",
        "num_solar_panels": 1500,
        "num_wind_turbines": 50,
        "battery_kwh": 2500
    }
    response = requests.post(url, json=payload)
    # The ML models need to be loaded, if they are not, this will return 503
    # Check for success, but also handle 503 if models haven't been trained/loaded yet
    if response.status_code == 503:
        print(
            f"\n⚠️ ML Prediction endpoint returned 503. This is expected if ML models haven't been trained by Airflow DAG yet.")
        pytest.skip("ML models not loaded, skipping ML prediction test.")

    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] is not None
    assert "total_cost" in data["predictions"]
    assert "self_sufficiency_pct" in data["predictions"]
    print(f"\n✅ /predict_ml endpoint returned valid predictions.")