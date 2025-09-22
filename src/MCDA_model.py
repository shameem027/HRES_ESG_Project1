# --- File: src/MCDA_model.py (Definitive Final Version - Robust Filtering) ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
# Import ALL relevant constants from the generator to use in logic and for passing to API
from src.HRES_Dataset_Generator import (
    COST_PER_SOLAR_PANEL, COST_PER_WIND_TURBINE, COST_PER_BATTERY_KWH,
    INSTALLATION_OVERHEAD_FACTOR, ENGINEERING_CONSULTING_COST_FACTOR, PERMITTING_LEGAL_COST_FACTOR,
    OTHER_COMPONENTS_COST_FACTOR, ANNUAL_OM_RATE, BATTERY_LIFETIME_YEARS,
    PROJECT_LIFETIME_YEARS, FINANCING_INTEREST_RATE
)

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:  # Prevent adding handlers multiple times if reloaded
    logging.basicConfig(level=logging.INFO)


def find_pareto_front(df: pd.DataFrame, objectives: dict):
    df_reset = df.reset_index(drop=True)
    is_pareto = np.ones(df_reset.shape[0], dtype=bool)
    for i, row in df_reset.iterrows():
        if not is_pareto[i]: continue
        for j, other_row in df_reset.iterrows():
            if i == j: continue
            is_dominated = all(
                (goal == 'maximize' and other_row[obj] >= row[obj]) or (
                        goal == 'minimize' and other_row[obj] <= row[obj])
                for obj, goal in objectives.items()
            ) and any(
                (goal == 'maximize' and other_row[obj] > row[obj]) or (goal == 'minimize' and other_row[obj] < row[obj])
                for obj, goal in objectives.items()
            )
            if is_dominated:
                is_pareto[i] = False
                break
    return df.iloc[df_reset[is_pareto].index]


class HRES_Decision_Engine:
    def __init__(self, configurations_df: pd.DataFrame):
        if configurations_df is None or configurations_df.empty:
            raise ValueError("Configuration DataFrame cannot be empty.")
        self.all_configs = configurations_df
        self.scaler = MinMaxScaler()
        self.esg_kpi_schema = {
            'env_co2_reduction_tons_yr': 'maximize', 'env_land_use_sqm': 'minimize',
            'env_water_savings_m3_yr': 'maximize', 'env_waste_factor_pct': 'minimize',
            'env_degradation_rate': 'minimize',
            'soc_local_jobs_fte': 'maximize', 'soc_energy_resilience_hrs': 'maximize',
            'soc_grid_strain_reduction_pct': 'maximize', 'soc_community_investment_eur': 'maximize',
            'soc_noise_level_impact_score': 'maximize',
            'gov_payback_plausibility_score': 'maximize', 'gov_supply_chain_transparency_score': 'maximize',
            'gov_regulatory_compliance_score': 'maximize', 'gov_stakeholder_reporting_score': 'maximize',
            'gov_operational_risk_score': 'maximize'
        }

    def _get_scaled_scenario_df(self, scenario_name, annual_demand_kwh):
        scenario_df = self.all_configs[self.all_configs['scenario_name'] == scenario_name].copy()
        if scenario_df.empty:
            logger.warning(f"Scenario '{scenario_name}' not found in dataset. Returning None.")
            return None

        original_demand = scenario_df['annual_demand_kwh'].iloc[0]
        scaling_factor = annual_demand_kwh / original_demand if original_demand > 0 else 0

        cols_to_scale = ['annual_kwh_generated', 'annual_savings_eur', 'total_cost', 'env_co2_reduction_tons_yr',
                         'soc_local_jobs_fte', 'annual_kwh_exported', 'annual_kwh_curtailed', 'env_water_savings_m3_yr',
                         'soc_community_investment_eur', 'wind_generation_kwh', 'solar_generation_kwh',
                         'annual_maintenance_cost_eur', 'annual_amortized_battery_replacement_cost_eur',
                         'annual_financing_cost_eur']  # All new cost columns must be scaled
        for col in cols_to_scale:
            if col in scenario_df.columns:
                scenario_df[col] *= scaling_factor
            else:
                logger.warning(
                    f"Column '{col}' not found in scenario_df for scaling. It might be missing from the generated dataset. Defaulting to 0.0.")
                scenario_df[col] = 0.0  # Add as 0 if missing for downstream calculations

        scenario_df['annual_demand_kwh'] = annual_demand_kwh
        grid_import = scenario_df['annual_demand_kwh'] - (
                scenario_df['annual_kwh_generated'] - scenario_df['annual_kwh_exported'])
        scenario_df['self_sufficiency_pct'] = ((annual_demand_kwh - np.maximum(0,
                                                                               grid_import)) / annual_demand_kwh) * 100
        scenario_df['self_sufficiency_pct'] = scenario_df['self_sufficiency_pct'].clip(upper=100.0)
        scenario_df['payback_period_years'] = scenario_df['total_cost'] / (scenario_df['annual_savings_eur'] + 1e-6)

        # Calculate wind contribution percentage
        total_gen_scaled = scenario_df['wind_generation_kwh'] + scenario_df['solar_generation_kwh']
        scenario_df['wind_contribution_pct'] = (scenario_df['wind_generation_kwh'] / (total_gen_scaled + 1e-9)) * 100
        scenario_df['wind_contribution_pct'] = scenario_df['wind_contribution_pct'].fillna(0).clip(upper=100.0)

        return scenario_df

    def step_1_moo_filter_feasible_solutions(self, scaled_scenario_df: pd.DataFrame, user_grid_dependency_pct: float):
        min_self_sufficiency = 100.0 - user_grid_dependency_pct
        min_wind_contribution_pct = 15.0  # User requested minimum 15% wind turbine contribution

        if 'wind_contribution_pct' not in scaled_scenario_df.columns:
            logger.error("'wind_contribution_pct' column missing from scaled_scenario_df. This is a critical error.")
            return None, "Internal error: Wind contribution data missing for filtering."

        feasible_solutions = scaled_scenario_df[
            (scaled_scenario_df['self_sufficiency_pct'] >= min_self_sufficiency) &
            (scaled_scenario_df['wind_contribution_pct'] >= min_wind_contribution_pct)
            ].copy()

        if feasible_solutions.empty:
            return None, f"No solutions meet the {min_self_sufficiency:.1f}% self-sufficiency target AND minimum {min_wind_contribution_pct:.1f}% wind contribution."

        feasible_solutions = feasible_solutions[feasible_solutions[
                                                    'payback_period_years'] <= PROJECT_LIFETIME_YEARS * 2].copy()  # Example: max 50 years payback

        if feasible_solutions.empty:
            return None, f"No solutions found with a reasonable payback period (max {PROJECT_LIFETIME_YEARS * 2} years) after meeting self-sufficiency and wind contribution targets."

        return feasible_solutions, f"Found {len(feasible_solutions)} technically feasible solutions."

    def step_2_electre_tri_sort_by_esg(self, feasible_df: pd.DataFrame):
        if feasible_df is None or feasible_df.empty: return feasible_df, "No feasible solutions to sort."
        profiles = {
            "Good": {'soc_energy_resilience_hrs': 8.0, 'gov_payback_plausibility_score': 7.0,
                     'env_co2_reduction_tons_yr_norm': 0.7},
            "Average": {'soc_energy_resilience_hrs': 4.0, 'gov_payback_plausibility_score': 5.0,
                        'env_co2_reduction_tons_yr_norm': 0.4}
        }
        df = feasible_df.copy()

        if 'env_co2_reduction_tons_yr' in df.columns and df['env_co2_reduction_tons_yr'].sum() > 0:
            df['env_co2_reduction_tons_yr_norm'] = self.scaler.fit_transform(df[['env_co2_reduction_tons_yr']])
        else:
            df['env_co2_reduction_tons_yr_norm'] = 0.0

        def assign_category(row):
            good_score = sum(1 for kpi, threshold in profiles["Good"].items() if row.get(kpi, 0) >= threshold)
            if good_score >= 2: return "Good"
            avg_score = sum(1 for kpi, threshold in profiles["Average"].items() if row.get(kpi, 0) >= threshold)
            if avg_score >= 2: return "Average"
            return "Poor"

        feasible_df['esg_category'] = df.apply(assign_category, axis=1)
        return feasible_df, "Sorted solutions into ESG categories."

    def step_3_mcda_select_best(self, sorted_df: pd.DataFrame, esg_weights: dict):
        if sorted_df is None or sorted_df.empty: return None, "No sorted solutions to select from."
        best_category_df = sorted_df[sorted_df['esg_category'] == 'Good']
        if best_category_df.empty: best_category_df = sorted_df[sorted_df['esg_category'] == 'Average']
        if best_category_df.empty: best_category_df = sorted_df
        mcda_df = best_category_df.copy()

        mcda_df['waste_factor'] = mcda_df['annual_kwh_curtailed'] / (mcda_df['annual_kwh_generated'] + 1e-6)

        cols_to_normalize = list(self.esg_kpi_schema.keys()) + ['total_cost', 'waste_factor']

        for col in cols_to_normalize:
            if col not in mcda_df.columns:
                logger.warning(
                    f"Column '{col}' not found in MCDA DataFrame for normalization. Adding with default 0.0.")
                mcda_df[col] = 0.0

        normalized_data = mcda_df[cols_to_normalize].copy()

        for col, goal in self.esg_kpi_schema.items():
            if col in normalized_data.columns and normalized_data[col].sum() > 0:
                if goal == 'minimize':
                    normalized_data[col] = 1 - self.scaler.fit_transform(normalized_data[[col]])
                else:
                    normalized_data[col] = self.scaler.fit_transform(normalized_data[[col]])
            elif col in normalized_data.columns:
                normalized_data[col] = 0.5

        for col in ['total_cost', 'waste_factor']:
            if col in normalized_data.columns and normalized_data[col].sum() > 0:
                normalized_data[col] = 1 - self.scaler.fit_transform(normalized_data[[col]])
            elif col in normalized_data.columns:
                normalized_data[col] = 0.5

        final_score = pd.Series(0.0, index=mcda_df.index)
        for dim, weight in esg_weights.items():
            if dim == 'cost':
                if 'total_cost' in normalized_data:
                    final_score += normalized_data['total_cost'] * weight
            else:
                dim_prefix = dim[:3]
                dim_cols = [col for col in self.esg_kpi_schema if col.startswith(dim_prefix + '_')]
                for col in dim_cols:
                    if col in normalized_data:
                        final_score += normalized_data[col] * (weight / (len(dim_cols) + 1e-6))

        if 'waste_factor' in normalized_data:
            final_score -= normalized_data['waste_factor'] * 0.5

        mcda_df['final_score'] = final_score

        if mcda_df.empty:
            return None, "No solutions remaining after MCDA scoring."

        best_solution_index = mcda_df['final_score'].idxmax()
        best_solution = sorted_df.loc[best_solution_index]
        return best_solution, f"Selected best solution from the '{best_solution['esg_category']}' ESG category."

    def run_full_pipeline(self, scenario_name, annual_demand_kwh, user_grid_dependency_pct, esg_weights):
        scaled_df = self._get_scaled_scenario_df(scenario_name, annual_demand_kwh)
        if scaled_df is None: return None, f"Scenario '{scenario_name}' not found.", None, None, None

        logger.info(
            f"Before filtering: {len(scaled_df)} solutions for {scenario_name} (Demand: {annual_demand_kwh:,} kWh).")
        logger.info(
            f"Targets: Max Grid Dependency: {user_grid_dependency_pct}%, Min Self-Sufficiency: {100.0 - user_grid_dependency_pct:.1f}%. Min Wind Contribution: 15.0%.")

        feasible_df, msg1 = self.step_1_moo_filter_feasible_solutions(scaled_df, user_grid_dependency_pct)
        if feasible_df is None:
            logger.warning(f"Step 1 Filter yielded no solutions for {scenario_name}: {msg1}")
            return None, msg1, None, None, None

        logger.info(f"After feasible filter: {len(feasible_df)} solutions. Message: {msg1}")

        sorted_df, msg2 = self.step_2_electre_tri_sort_by_esg(feasible_df)

        logger.info(f"After ESG sort: {len(sorted_df)} solutions. Message: {msg2}")

        best_solution, msg3 = self.step_3_mcda_select_best(sorted_df, esg_weights)
        pareto_front_df = find_pareto_front(feasible_df.copy(),
                                            {'total_cost': 'minimize', 'self_sufficiency_pct': 'maximize'})

        # Add all relevant model constants to the best_solution dictionary for UI to access
        if best_solution is not None:
            best_solution['model_constants'] = {
                'COST_PER_SOLAR_PANEL': COST_PER_SOLAR_PANEL,
                'COST_PER_WIND_TURBINE': COST_PER_WIND_TURBINE,
                'COST_PER_BATTERY_KWH': COST_PER_BATTERY_KWH,
                'INSTALLATION_OVERHEAD_FACTOR': INSTALLATION_OVERHEAD_FACTOR,
                'ENGINEERING_CONSULTING_COST_FACTOR': ENGINEERING_CONSULTING_COST_FACTOR,
                'PERMITTING_LEGAL_COST_FACTOR': PERMITTING_LEGAL_COST_FACTOR,
                'OTHER_COMPONENTS_COST_FACTOR': OTHER_COMPONENTS_COST_FACTOR,
                'ANNUAL_OM_RATE': ANNUAL_OM_RATE,
                'BATTERY_LIFETIME_YEARS': BATTERY_LIFETIME_YEARS,
                'PROJECT_LIFETIME_YEARS': PROJECT_LIFETIME_YEARS,
                'FINANCING_INTEREST_RATE': FINANCING_INTEREST_RATE
            }
        return best_solution, f"{msg1} {msg2} {msg3}", feasible_df, sorted_df, pareto_front_df