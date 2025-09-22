# --- File: tests/test_infrastructure.py (Placeholder) ---
import pytest
import os
import subprocess

# Define the root of your project for path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_requirements_txt_exists():
    """Test that requirements.txt exists in the project root."""
    assert os.path.exists(os.path.join(PROJECT_ROOT, 'requirements.txt')), \
        "requirements.txt not found in project root!"


def test_docker_compose_config_valid():
    """Test that the docker-compose.yml file is syntactically valid."""
    docker_compose_path = os.path.join(PROJECT_ROOT, 'docker', 'docker-compose.yml')
    assert os.path.exists(docker_compose_path), \
        f"docker-compose.yml not found at {docker_compose_path}"

    try:
        # Run 'docker compose config' to validate syntax
        subprocess.run(['docker', 'compose', '-f', docker_compose_path, 'config'],
                       check=True,
                       capture_output=True,
                       text=True)
        print(f"\n✅ Docker Compose configuration for {docker_compose_path} is valid.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"❌ Docker Compose configuration is invalid: {e.stderr}")


def test_essential_dockerfiles_exist():
    """Verify that all expected Dockerfiles exist."""
    expected_dockerfiles = [
        'Dockerfile.airflow',
        'Dockerfile.api',
        'Dockerfile.jupyter',
        'Dockerfile.mlflow',
        'Dockerfile.ui'
    ]
    for df in expected_dockerfiles:
        path = os.path.join(PROJECT_ROOT, 'docker', df)
        assert os.path.exists(path), f"❌ Missing Dockerfile: {path}"
        print(f"✅ Dockerfile exists: {path}")


# This test requires the Docker stack to be running, typically for integration tests
# It's here as an example for the CI/CD but might be in a separate integration test suite
def test_full_stack_health_check():
    """
    (Placeholder for integration test) - Checks if key services are reachable.
    This test assumes services are already running (e.g., from a 'deploy' step).
    """
    api_health_url = "http://localhost:8081/health"  # Assuming API is mapped to host port 8081
    try:
        response = requests.get(api_health_url, timeout=10)
        response.raise_for_status()
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
        print(f"\n✅ HRES API at {api_health_url} is healthy.")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"❌ HRES API health check failed: {e}")

# You can add more tests here, e.g.,
# def test_src_directory_exists():
#     assert os.path.exists(os.path.join(PROJECT_ROOT, 'src'))

# def test_api_directory_exists():
#     assert os.path.exists(os.path.join(PROJECT_ROOT, 'api'))