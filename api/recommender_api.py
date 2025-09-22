# --- File: api/recommender_api.py (Definitive Final Version - ML Prediction Endpoint) ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: Serves the HRES Decision Engine via a structured API and a conversational LLM endpoint ---

import os
import pandas as pd
from flask import Flask, request, jsonify
import logging
import json
from openai import AzureOpenAI
import mlflow
import mlflow.sklearn

# -----------------------------------------------------------------------------
# 1. SETUP AND INITIALIZATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.MCDA_model import HRES_Decision_Engine  # Ensure MCDA_model is imported here
from src.HRES_ML_Model import HRESMLPredictor  # Import ML Predictor class

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 2. LOAD RESOURCES ON STARTUP
# -----------------------------------------------------------------------------
decision_engine = None
try:
    DATA_PATH = "/app/src/HRES_Dataset.csv"
    hres_df = pd.read_csv(DATA_PATH)
    decision_engine = HRES_Decision_Engine(hres_df)
    logger.info(f"✅ HRES Decision Engine initialized successfully with {len(hres_df)} configurations.")
except FileNotFoundError:
    logger.error(f"❌ CRITICAL: The dataset file was not found at {DATA_PATH}. The API will not be functional.")
    logger.error(f"Please run 'docker-compose run --rm jupyter python src/HRES_Dataset_Generator.py' to create it.")
except Exception as e:
    logger.error(f"❌ CRITICAL: Could not initialize Decision Engine. Error: {e}", exc_info=True)

azure_client = None
try:
    azure_endpoint_env = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint_env and not azure_endpoint_env.endswith("/"):
        azure_endpoint_env += "/"  # Ensure trailing slash for consistency

    azure_client = AzureOpenAI(
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=azure_endpoint_env,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    logger.info("✅ Azure OpenAI client initialized.")
except Exception as e:
    logger.warning(
        f"⚠️ Could not initialize Azure OpenAI client. Chat functionality disabled. Error: {e}. Check your .env file for correct Azure OpenAI credentials and format (e.g., endpoint with trailing slash).")

# -------------------- ML Model Loading --------------------
ml_models = {}
scenario_encoders = {}
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://hres_mlflow:5000")  # Use hres_mlflow service name
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load latest ML models for prediction targets
for target in ['total_cost', 'self_sufficiency_pct', 'annual_savings_eur']:
    model, encoder = HRESMLPredictor.load_latest_model(target, model_name_suffix="_V1")
    if model and encoder:
        ml_models[target] = model
        scenario_encoders[target] = encoder
    else:
        logger.error(f"Failed to load ML model for {target}. ML prediction endpoint may not function correctly.")
logger.info(f"Loaded {len(ml_models)} ML models.")
# ----------------------------------------------------------

# -----------------------------------------------------------------------------
# 3. DEFINE LLM PROMPTS (PhD-Level for accuracy)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = "You are an expert AI consultant for ESG-integrated renewable energy systems. Your tone is professional, helpful and data-driven. Strictly adhere to the requested JSON output format."

INTENT_PARSING_PROMPT = """
You are a precision AI assistant for a renewable energy consulting firm. Your sole task is to extract structured data from a user's request.
You MUST output ONLY a single, valid JSON object, and it MUST contain ALL the following keys.
If a value is not explicitly provided by the user, you MUST infer a reasonable default based on the common usage or the scenario type.

1.  "scenario_name": (string) The building type. MUST be one of ["Small_Office", "University_Campus", "Hospital", "Industrial_Facility", "Data_Center"]. Infer the closest match.
    *   Examples: "office" -> "Small_Office", "university" -> "University_Campus", "data center" -> "Data_Center".

2.  "annual_demand_kwh": (integer) The annual electricity consumption in kWh.
    *   If not explicitly provided, use these realistic defaults:
        *   "Small_Office": 250000
        *   "University_Campus": 3000000
        *   "Hospital": 1500000
        *   "Industrial_Facility": 5000000
        *   "Data_Center": 10000000

3.  "user_grid_dependency_pct": (integer, 0-100) The maximum allowed grid dependency. Lower is more self-sufficient.
    *   "off-grid" or "fully independent" -> 0
    *   "highly self-sufficient" or "minimal grid reliance" -> 10
    *   "mostly self-sufficient" -> 15
    *   If not mentioned, default to 30.

4.  "esg_weights": (object) A dictionary with keys "environment", "social", "governance", "cost".
    *   The sum of the values MUST be exactly 1.0 (normalize if necessary after inference).
    *   Infer priorities from keywords. Default to a balanced split (0.25 for each) if no strong priorities are given.
    *   Keywords for weights:
        *   "low cost", "cheap", "cost-effective": higher "cost" weight.
        *   "green", "environmental impact", "sustainable": higher "environment" weight.
        *   "social good", "community benefits", "local jobs": higher "social" weight.
        *   "governance", "responsible", "transparency": higher "governance" weight.
    *   If multiple priorities: distribute weights accordingly, ensuring they sum to 1.0.

User Query: "{query}"

Examples of desired JSON output (ALWAYS complete, with all keys):
- User: "find a solution for a small office"
  JSON: {{"scenario_name": "Small_Office", "annual_demand_kwh": 250000, "user_grid_dependency_pct": 30, "esg_weights": {{"environment": 0.25, "social": 0.25, "governance": 0.25, "cost": 0.25}}}}

- User: "I need a low cost solution for a small office with strong environmental impact"
  JSON: {{"scenario_name": "Small_Office", "annual_demand_kwh": 250000, "user_grid_dependency_pct": 30, "esg_weights": {{"environment": 0.4, "social": 0.1, "governance": 0.1, "cost": 0.4}}}}

- User: "hospital, 1.2M kWh demand, high resilience, focus on social good"
  JSON: {{"scenario_name": "Hospital", "annual_demand_kwh": 1200000, "user_grid_dependency_pct": 10, "esg_weights": {{"environment": 0.2, "social": 0.6, "governance": 0.1, "cost": 0.1}}}}

- User: "university with 3 million kWh, cost is top priority"
  JSON: {{"scenario_name": "University_Campus", "annual_demand_kwh": 3000000, "user_grid_dependency_pct": 30, "esg_weights": {{"environment": 0.1, "social": 0.1, "governance": 0.1, "cost": 0.7}}}}

JSON Output:
"""

RESPONSE_GENERATION_PROMPT = """
You are an expert AI consultant for ESG-integrated renewable energy systems, acting as an advisor for the HRES Decision Support Platform.
Your task is to generate a comprehensive, professional, and visually appealing report based on a quantitative model's recommendation.
You MUST use ONLY the data provided in the 'CONTEXT'. Do not invent or infer any data.
Your response MUST be a single, valid JSON object with two keys: "summary" (a concise, professional paragraph) and "details" (an array of objects, each with "title" and "content" keys).
The "content" must be well-formatted Markdown. Use bullet points, bolding, and create Markdown tables for clarity. For all numbers, use comma separators for thousands. All currency is in Euros (€).

CONTEXT:
{context}

JSON Response:
"""


# -----------------------------------------------------------------------------
# 4. DEFINE API ENDPOINTS
# -----------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Provides a simple health check for the service."""
    return jsonify({
        "status": "healthy",
        "decision_engine_loaded": decision_engine is not None,
        "llm_client_loaded": azure_client is not None,
        "ml_models_loaded": bool(ml_models)
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for the structured, quantitative recommender UI."""
    if not decision_engine:
        return jsonify(
            {"error": "Decision Engine is not operational. Check server startup logs for a critical error."}), 503

    try:
        data = request.get_json()
        best_solution, status_message, feasible_df, sorted_df, pareto_front_df = decision_engine.run_full_pipeline(
            data['scenario_name'], data['annual_demand_kwh'], data['user_grid_dependency_pct'], data['esg_weights']
        )

        if best_solution is not None:
            response = {
                "status": status_message,
                "recommendation": best_solution.to_dict(),  # model_constants are added in MCDA_model.py
                "intermediate_results": {
                    "feasible_count": len(feasible_df) if feasible_df is not None else 0,
                    "esg_category_counts": sorted_df[
                        'esg_category'].value_counts().to_dict() if sorted_df is not None and 'esg_category' in sorted_df else {},
                    "pareto_front": pareto_front_df.to_dict(orient='records') if pareto_front_df is not None else []
                }
            }
            return jsonify(response), 200
        else:
            logger.warning(f"No solution found for /recommend request. Status: {status_message}")
            return jsonify({"status": status_message, "recommendation": None, "intermediate_results": {}}), 404

    except KeyError as e:
        logger.error(f"Missing key in request for /recommend: {e}")
        return jsonify({"error": f"Missing required field in request: {e}"}), 400
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint for the conversational advisor UI."""
    if not decision_engine:
        return jsonify({"error": "Decision Engine is not operational. Check API logs for critical errors."}), 503
    if not azure_client:
        return jsonify({
                           "error": "Conversational AI is disabled. Azure OpenAI client could not be initialized. Check .env file for valid credentials."}), 503

    try:
        data = request.get_json()
        user_query = data['query']

        # --- LLM Call 1: Parse User Intent ---
        parsed_intent = None
        try:
            intent_prompt = INTENT_PARSING_PROMPT.format(query=user_query)
            logger.info(f"Sending to LLM (Intent Parsing): {intent_prompt}")  # Debug logging

            intent_response = azure_client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.0,  # Keep temperature low for deterministic parsing
                response_format={"type": "json_object"}
            )
            raw_llm_output = intent_response.choices[0].message.content
            logger.info(f"Raw LLM output (Intent Parsing): {raw_llm_output}")  # Debug logging

            parsed_intent = json.loads(raw_llm_output)
            logger.info(f"Parsed user intent: {parsed_intent}")

            # Validate required keys after JSON parsing
            required_keys = ['scenario_name', 'annual_demand_kwh', 'user_grid_dependency_pct', 'esg_weights']
            if not all(k in parsed_intent for k in required_keys):
                raise KeyError(
                    f"LLM output missing one of the required keys: {required_keys}. Raw output: {raw_llm_output}")

            # Basic validation and normalization of esg_weights structure
            if not isinstance(parsed_intent['esg_weights'], dict) or \
                    not all(k in parsed_intent['esg_weights'] for k in ["environment", "social", "governance", "cost"]):
                raise ValueError(f"ESG weights format invalid. Raw output: {raw_llm_output}")

            # Normalize ESG weights if they don't sum to 1.0 (due to LLM approximation)
            current_sum = sum(parsed_intent['esg_weights'].values())
            if not abs(current_sum - 1.0) < 0.01:
                logger.warning(f"LLM generated ESG weights sum to {current_sum:.2f}, normalizing to 1.0.")
                normalized_weights = {k: v / current_sum for k, v in parsed_intent['esg_weights'].items()}
                parsed_intent['esg_weights'] = normalized_weights
                logger.info(f"Normalized ESG weights: {parsed_intent['esg_weights']}")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"LLM output parsing or validation failed: {e}", exc_info=True)
            ai_response = {
                "summary": "I'm having trouble understanding your request. It seems some key information (like building type, energy demand, or your priorities) is missing or unclear, or the AI's response was not in the expected format. Please try rephrasing more explicitly, for example: 'Find a low-cost solution for a small office with 250,000 kWh annual demand and a strong environmental focus.'",
                "details": []}
            return jsonify({"response": ai_response})

        # --- Run Quantitative Model (The "R" in RAG) ---
        best_solution, _, _, _, _ = decision_engine.run_full_pipeline(
            parsed_intent['scenario_name'], parsed_intent['annual_demand_kwh'],
            parsed_intent['user_grid_dependency_pct'], parsed_intent['esg_weights']
        )

        if best_solution is None:
            ai_response = {
                "summary": "Based on my interpretation of your request, our quantitative model could not find a feasible solution. This usually means the goals are too ambitious for the building type. Please try relaxing your constraints (e.g., allow more grid dependency).",
                "details": []}
        else:
            # --- LLM Call 2: Generate Structured Explanation (The "G" in RAG) ---
            # Construct context for LLM to generate comprehensive report
            context_data = best_solution.to_dict()
            context_data['user_query'] = user_query
            context_data['parsed_inputs'] = parsed_intent

            # The model_constants are already part of best_solution now.

            response_prompt = RESPONSE_GENERATION_PROMPT.format(context=json.dumps(context_data, indent=2))

            final_response = azure_client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": response_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            ai_response = json.loads(final_response.choices[0].message.content)

        return jsonify({"response": ai_response})

    except Exception as e:
        logger.error(f"Critical error in /chat endpoint: {e}", exc_info=True)
        return jsonify({
                           "error": "An unexpected error occurred while processing your request with the AI. Check API logs for details."}), 500


@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    """New endpoint for ML model predictions for fast inference."""
    if not ml_models or not scenario_encoders:
        return jsonify(
            {"error": "ML models not loaded or scenario encoder missing. Please train models via Airflow DAG."}), 503

    try:
        data = request.get_json()
        scenario_name = data.get('scenario_name')
        num_solar_panels = data.get('num_solar_panels')
        num_wind_turbines = data.get('num_wind_turbines')
        battery_kwh = data.get('battery_kwh')

        if not all([scenario_name, num_solar_panels, num_wind_turbines, battery_kwh]) or \
                not isinstance(num_solar_panels, (int, float)) or \
                not isinstance(num_wind_turbines, (int, float)) or \
                not isinstance(battery_kwh, (int, float)):
            return jsonify({
                               "error": "Missing or invalid input for ML prediction. Required: scenario_name (str), num_solar_panels (int), num_wind_turbines (int), battery_kwh (int)."}), 400

        # Prepare features for prediction
        if scenario_name not in scenario_encoders['total_cost']:  # Assuming all encoders are same
            return jsonify({
                               "error": f"Scenario '{scenario_name}' not recognized by ML model. Available scenarios: {list(scenario_encoders['total_cost'].keys())}"}), 400

        scenario_encoded = scenario_encoders['total_cost'][scenario_name]  # Use any encoder, they should be the same

        input_df = pd.DataFrame([[num_solar_panels, num_wind_turbines, battery_kwh, scenario_encoded]],
                                columns=['num_solar_panels', 'num_wind_turbines', 'battery_kwh', 'scenario_encoded'])

        predictions = {}
        for target, model in ml_models.items():
            predictions[target] = round(model.predict(input_df)[0], 2)  # Round predictions

        return jsonify({"status": "ML prediction successful", "predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Error in /predict_ml endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during ML prediction: {e}"}), 500


# -----------------------------------------------------------------------------
# 5. START THE APPLICATION
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)