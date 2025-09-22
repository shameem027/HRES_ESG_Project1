# --- File: src/HRES_ML_Model.py (NEW FILE - Robust ML Model for Prediction) ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: Trains and logs Machine Learning models to predict HRES outcomes for faster inference. ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://hres_mlflow:5000")  # Use hres_mlflow service name
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("HRES_ML_Prediction_Models")


class HRESMLPredictor:
    def __init__(self, model_name_suffix=""):
        self.models = {}
        self.model_name_suffix = model_name_suffix
        # The target columns we want to predict
        self.target_cols = ['total_cost', 'self_sufficiency_pct', 'annual_savings_eur']
        # The features used for prediction (inputs to the ML model)
        self.feature_cols = ['num_solar_panels', 'num_wind_turbines', 'battery_kwh']
        self.scenario_encoder = {}  # To store scenario encoding mapping

    def _prepare_data(self, df):
        # One-hot encode 'scenario_name' to include it as a feature
        scenario_names = df['scenario_name'].unique()
        self.scenario_encoder = {name: i for i, name in enumerate(scenario_names)}

        df_encoded = df.copy()
        # Add the encoded scenario as a numerical feature
        df_encoded['scenario_encoded'] = df_encoded['scenario_name'].map(self.scenario_encoder)

        # Combine original features with the new encoded scenario feature
        features = self.feature_cols + ['scenario_encoded']

        X = df_encoded[features]
        y = df_encoded[self.target_cols]
        return X, y

    def train_and_log_models(self, dataset_path: str):
        df = pd.read_csv(dataset_path)
        logger.info(f"Training ML models on dataset with {len(df)} rows.")

        X, y = self._prepare_data(df)

        for target in self.target_cols:
            with mlflow.start_run(run_name=f"RandomForest_{target}_{self.model_name_suffix}") as run:
                logger.info(f"Training model for target: {target}")

                # Model parameters (can be tuned further for better performance)
                params = {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "min_samples_split": 5,
                    "min_samples_leaf": 3,
                }

                model = RandomForestRegressor(**params)
                model.fit(X, y[target])

                y_pred = model.predict(X)
                mae = mean_absolute_error(y[target], y_pred)
                r2 = r2_score(y[target], y_pred)

                logger.info(f"Target: {target}, MAE: {mae:.2f}, R2: {r2:.2f}")

                mlflow.log_params(params)
                mlflow.log_metrics({"mae": mae, "r2": r2})

                # Log and register the model to MLflow Model Registry
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"hres_ml_model_{target}",  # Path within the MLflow run artifact storage
                    registered_model_name=f"HRES_ML_Predictor_{target}{self.model_name_suffix}"
                    # Name in Model Registry
                )

                # Log scenario_encoder as a JSON artifact for consistent loading
                mlflow.log_dict(self.scenario_encoder, "scenario_encoder.json")

                self.models[target] = model  # Store model in instance

        logger.info("ML models training and logging complete.")

    @staticmethod
    def load_latest_model(target: str, model_name_suffix=""):
        model_name = f"HRES_ML_Predictor_{target}{model_name_suffix}"

        # Load the latest version of the model from MLflow Model Registry
        client = mlflow.tracking.MlflowClient()
        try:
            # Get the latest version object for the model
            latest_version_obj = client.get_latest_versions(model_name, stages=["None"])[0]
            latest_version = latest_version_obj.version
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            logger.info(f"Successfully loaded ML model '{model_name}' v{latest_version}")

            # Retrieve scenario_encoder associated with this model run
            run_id = latest_version_obj.run_id
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/scenario_encoder.json",
                                                             dst_path="/tmp")
            with open(os.path.join(local_path, "scenario_encoder.json"), 'r') as f:
                scenario_encoder = json.load(f)
            logger.info(f"Loaded scenario encoder: {scenario_encoder}")

            return model, scenario_encoder
        except Exception as e:
            logger.error(f"Failed to load ML model '{model_name}': {e}", exc_info=True)
            return None, None


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'HRES_Dataset.csv')
    predictor = HRESMLPredictor(model_name_suffix="_V1")
    predictor.train_and_log_models(dataset_path)


if __name__ == "__main__":
    main()