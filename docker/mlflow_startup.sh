#!/bin/bash
# --- File: docker/mlflow_startup.sh (Definitive Final Version - Robust MLflow Startup Script) ---
set -e # Exit immediately if a command exits with a non-zero status

echo "Starting MLflow robust startup sequence within custom image..."

# Export environment variables for MLflow to ensure they are recognized
export MLFLOW_TRACKING_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@hres_postgres:5432/${POSTGRES_DB}"
export MLFLOW_BACKEND_STORE_URI=${MLFLOW_TRACKING_URI}
export MLFLOW_DEFAULT_ARTIFACT_ROOT="/mlflow/artifacts"

echo "MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
echo "MLFLOW_BACKEND_STORE_URI: ${MLFLOW_BACKEND_STORE_URI}"
echo "MLFLOW_DEFAULT_ARTIFACT_ROOT: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"

echo "Waiting for Postgres to be fully ready before MLflow DB upgrade..."
until pg_isready -h hres_postgres -p 5432 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"; do
  echo "Postgres not yet ready. Sleeping for 2 seconds..."
  sleep 2
done
echo "Postgres is ready. Running MLflow DB upgrade."

upgrade_db() {
    mlflow db upgrade ${MLFLOW_BACKEND_STORE_URI}
}

MAX_RETRIES=10
RETRY_COUNT=0
until upgrade_db || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    echo "MLflow DB upgrade failed, retrying in 5 seconds... ($((MAX_RETRIES-RETRY_COUNT)) retries left)"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "MLflow DB upgrade failed after $MAX_RETRIES attempts. Exiting."
    exit 1
fi

echo "MLflow DB upgrade complete. Starting MLflow server."

exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000