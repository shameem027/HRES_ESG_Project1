# --- File: tests/test_mcda_validation.py (Placeholder) ---
import pytest
import pandas as pd
import os
import sys

# Add the src directory to Python's path to allow importing our custom modules
# This path is relative to the project root where tests are run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from MCDA_model import HRES_Decision_Engine

# Define the path to the generated dataset
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'HRES_Dataset.csv'))


@pytest.fixture(scope="module")
def decision_engine_instance():
    """Fixture to load the dataset and initialize the decision engine once per test module."""
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset not found at {DATASET_PATH}. Run HRES_Dataset_Generator.py first.")

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        pytest.fail("Dataset is empty, cannot initialize Decision Engine.")
    return HRES_Decision_Engine(df)


def test_mcda_finds_solution_for_balanced_university(decision_engine_instance):
    """Test if the MCDA model can find a solution for a balanced university scenario."""
    if decision_engine_instance is None:
        pytest.skip("Decision Engine not initialized due to missing dataset.")

    scenario_name = "University_Campus"
    annual_demand_kwh = 3000000
    user_grid_dependency_pct = 20  # Allow some grid dependency
    esg_weights = {"environment": 0.25, "social": 0.25, "governance": 0.25, "cost": 0.25}

    best_solution, status_message, _, _, _ = decision_engine_instance.run_full_pipeline(
        scenario_name, annual_demand_kwh, user_grid_dependency_pct, esg_weights
    )

    assert best_solution is not None, f"MCDA failed to find a solution for balanced university: {status_message}"
    assert best_solution['self_sufficiency_pct'] >= (100 - user_grid_dependency_pct), "Self-sufficiency target not met."
    assert best_solution['payback_period_years'] > 0, "Payback period should be positive."
    print(
        f"\n✅ MCDA found a solution for balanced university. Total Cost: €{best_solution['total_cost']:,}, SS: {best_solution['self_sufficiency_pct']:.1f}%")


def test_mcda_enforces_min_wind_contribution(decision_engine_instance):
    """Test if the 15% minimum wind contribution is enforced."""
    if decision_engine_instance is None:
        pytest.skip("Decision Engine not initialized due to missing dataset.")

    scenario_name = "Small_Office"
    annual_demand_kwh = 250000
    user_grid_dependency_pct = 50  # Very high grid dependency, making it easier to find solutions
    esg_weights = {"environment": 0.5, "social": 0.2, "governance": 0.1, "cost": 0.2}

    best_solution, status_message, _, _, _ = decision_engine_instance.run_full_pipeline(
        scenario_name, annual_demand_kwh, user_grid_dependency_pct, esg_weights
    )

    assert best_solution is not None, f"MCDA failed to find a solution for min wind test: {status_message}"
    assert best_solution['wind_contribution_pct'] >= 15.0, "Minimum 15% wind contribution not enforced."
    print(f"\n✅ MCDA enforces min wind contribution. Wind: {best_solution['wind_contribution_pct']:.1f}%")


def test_mcda_returns_none_for_impossible_targets(decision_engine_instance):
    """Test if MCDA correctly returns None for impossible targets (e.g., 0% grid dependency with no feasible solution)."""
    if decision_engine_instance is None:
        pytest.skip("Decision Engine not initialized due to missing dataset.")

    # Attempt a very strict scenario that might have no solution
    scenario_name = "Data_Center"
    annual_demand_kwh = 10000000
    user_grid_dependency_pct = 0  # 100% self-sufficiency, strict!
    esg_weights = {"environment": 0.3, "social": 0.3, "governance": 0.2, "cost": 0.2}

    best_solution, status_message, _, _, _ = decision_engine_instance.run_full_pipeline(
        scenario_name, annual_demand_kwh, user_grid_dependency_pct, esg_weights
    )

    # It might be possible to find 100% self-sufficiency with the expanded dataset,
    # so we assert it either finds one or correctly explains why it didn't.
    if best_solution is None:
        assert "No solutions meet the" in status_message or "No solutions found with a reasonable payback period" in status_message
        print(
            f"\n✅ MCDA correctly indicated no solution for very strict Data Center (0% grid dependency). Status: {status_message}")
    else:
        assert best_solution['self_sufficiency_pct'] >= 100.0
        print(
            f"\n✅ MCDA found a 100% self-sufficient solution for Data Center! Total Cost: €{best_solution['total_cost']:,}")