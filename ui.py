# --- File: ui.py (PhD-Level State-of-the-Art Definitive Version - ML Prediction UI & Comprehensive Display) ---
# --- Author: Md Shameem Hossain ---
# --- Purpose: A professional Streamlit UI for the HRES Decision Support Platform ---

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HRES Advisor | ESG & Renewables Platform",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a more polished and modern look
st.markdown("""
<style>
    .stMetric {
        border-left: 5px solid #0068C9;
        padding: 0.5rem; /* Reduced padding */
        border-radius: 0.5rem;
        background-color: #F0F2F6;
    }
    .stButton>button {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    /* --- FIX: Reduce spacing in sidebar inputs --- */
    .stSelectbox, .stNumberInput, .stSlider, .stRadio, .stCheckbox { /* Added stRadio, stCheckbox for completeness */
        margin-bottom: 0.2rem; /* Reduce bottom margin */
        padding-bottom: 0.2rem; /* Reduce padding below */
    }
    div[data-testid="stVerticalBlock"] > div { /* Target vertical blocks for more compact layout */
        gap: 0.2rem; /* Reduce gap between elements in vertical blocks */
    }
    /* Specific targets for smaller gaps in sidebar sections */
    div[data-testid="stSidebar"] div.stVerticalBlock {
        gap: 0.2rem;
    }
    div[data-testid="stSidebar"] .stForm > div > div {
        gap: 0.2rem;
    }
    /* Ensure markdown lists in expanders are compact */
    .streamlit-expanderContent p {
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. API CONFIGURATION & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
RECOMMEND_API_URL = "http://hres_api:8081/recommend"  # Use the new host port 8081
CHAT_API_URL = "http://hres_api:8081/chat"  # Use the new host port 8081
ML_PREDICT_API_URL = "http://hres_api:8081/predict_ml"  # Use the new host port 8081


@st.cache_data(show_spinner=False)
def call_api(url, payload):
    """Cached function to call the backend API and handle errors."""
    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"**API Connection Error:** Could not connect to the backend service.")
        st.info(
            f"Please ensure the Docker environment is running correctly and the API is accessible on port 8081. Details: {e}")
        return None


# ==============================================================================
# 3. UI LAYOUT & COMPONENTS
# ==============================================================================

# --- Main Title ---
st.title("💡 HRES Decision Support Platform")
st.caption(
    f"An Interactive MLOps Platform for Modeling ESG-Integrated Hybrid Renewable Energy Systems | PhD Framework by **Md Shameem Hossain**")

# --- UI Tabs for Different Interaction Modes ---
tab1, tab2, tab3 = st.tabs(
    ["📊 **Quantitative Analysis Dashboard**", "💬 **Conversational AI Advisor**", "⚡ **ML Fast Predictor**"])

# ==============================================================================
# TAB 1: QUANTITATIVE ANALYSIS DASHBOARD
# ==============================================================================
with tab1:
    st.header("Structured Decision & Simulation")
    st.markdown(
        "Define your project parameters in the sidebar to run a full simulation. The system will execute the **MOO → ELECTRE TRI → MCDA** pipeline to find and justify the optimal HRES configuration.")
    st.divider()

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        if os.path.exists('logo.png'):
            st.image("logo.png", width=100)
        st.title("Project Controls")

        st.header("1. Scenario Definition")
        scenario_options = {
            "🏢 Small Office": ("Small_Office", 250000), "🎓 University Campus": ("University_Campus", 3000000),
            "🏥 Hospital": ("Hospital", 1500000), "🏭 Industrial Facility": ("Industrial_Facility", 5000000),
            "💻 Data Center": ("Data_Center", 10000000)
        }
        selected_scenario_display = st.selectbox("Building Type:", options=list(scenario_options.keys()), index=1)
        scenario, default_demand = scenario_options[selected_scenario_display]

        annual_demand = st.number_input("Annual Electricity Demand (kWh):", min_value=50000, value=default_demand,
                                        step=50000)
        grid_dependency = st.slider("Maximum Grid Dependency (%):", 0, 100, 30, help="Lower is more self-sufficient.")

        st.header("2. Decision Priorities")
        st.caption("Adjust the weights to reflect your project's primary goals. Total must be 1.0.")
        env_weight = st.slider("🌱 Environmental Weight", 0.0, 1.0, 0.4, 0.05)
        soc_weight = st.slider("👥 Social Weight", 0.0, 1.0, 0.3, 0.05)
        gov_weight = st.slider("🏛️ Governance Weight", 0.0, 1.0, 0.1, 0.05)
        cost_weight = st.slider("💰 Cost-Effectiveness Weight", 0.0, 1.0, 0.2, 0.05)

        total_weight = env_weight + soc_weight + gov_weight + cost_weight
        st.metric(label="Total Weight", value=f"{total_weight:.2f}")

        st.divider()
        find_button = st.button("Run Full Analysis", use_container_width=True, type="primary")

    # --- Main Content Area for Results ---
    if find_button:
        if not abs(total_weight - 1.0) < 0.01:
            st.error("Validation Error: The sum of all priority weights must be exactly 1.0.")
        else:
            payload = {
                "scenario_name": scenario, "annual_demand_kwh": annual_demand,
                "user_grid_dependency_pct": grid_dependency,
                "esg_weights": {"environment": env_weight, "social": soc_weight, "governance": gov_weight,
                                "cost": cost_weight}
            }
            with st.spinner("Executing decision pipeline... This may take a moment."):
                result = call_api(RECOMMEND_API_URL, payload)

            if result and result.get("recommendation"):
                rec = result["recommendation"]
                intermediate = result["intermediate_results"]
                # FIX: Extract model_constants from the API response
                model_constants = rec.pop('model_constants', {})  # Remove it from 'rec' so it's not displayed as a KPI

                st.success("Analysis Complete!", icon="✅")
                st.header("🏆 Executive Summary: Your Optimal HRES Solution")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Cost (CAPEX)", f"€{int(rec['total_cost']):,}")
                col2.metric("Self-Sufficiency", f"{rec['self_sufficiency_pct']:.1f}%")
                col3.metric("Payback Period", f"{rec['payback_period_years']:.1f} years")
                col4.metric("Annual Savings", f"€{int(rec['annual_savings_eur']):,}")

                # --- New Tabs for More Charts/Tables (10+ items) ---
                viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs(  # Added new tab for constants
                    ["🔬 **Pareto Front Analysis**", "🌱 **ESG KPI Deep Dive**", "⚡ **Energy Flow & Performance**",
                     "💶 **Detailed Cost Breakdown**", "⚙️ **Model Constants**"])

                with viz_tab1:
                    st.subheader("The Cost vs. Self-Sufficiency Trade-off")
                    st.caption(
                        "This chart shows all technically feasible solutions (the simulated Pareto front). Your recommendation is the one on this 'efficiency frontier' that best matches your ESG and cost priorities.")
                    pareto_df = pd.DataFrame(intermediate['pareto_front'])
                    if not pareto_df.empty:
                        fig = px.scatter(
                            pareto_df, x='total_cost', y='self_sufficiency_pct',
                            size='annual_savings_eur', color='esg_category',
                            hover_name='esg_category',
                            hover_data=['num_solar_panels', 'num_wind_turbines', 'battery_kwh', 'annual_savings_eur'],
                            labels={'total_cost': 'Total System Cost (€)',
                                    'self_sufficiency_pct': 'Energy Self-Sufficiency (%)',
                                    'esg_category': 'ESG Category'},
                            title="All Feasible HRES Configurations"
                        )
                        fig.add_scatter(x=[rec['total_cost']], y=[rec['self_sufficiency_pct']], mode='markers',
                                        marker=dict(color='red', size=20, symbol='star',
                                                    line=dict(color='black', width=2)), name='Your Recommendation')
                        st.plotly_chart(fig, use_container_width=True)

                with viz_tab2:
                    st.subheader("ESG KPI Deep Dive for Recommended Solution")
                    # KPI Table 1: Full ESG KPI Breakdown (Corrected)
                    kpi_data = {
                        'Dimension': [
                            'Environmental', 'Environmental', 'Environmental', 'Environmental', 'Environmental',
                            'Social', 'Social', 'Social', 'Social', 'Social',
                            'Governance', 'Governance', 'Governance', 'Governance', 'Governance'
                        ],
                        'KPI': [
                            'CO₂ Reduction (tons/yr)', 'Land Use (m²)', 'Water Savings (m³/yr)', 'Waste Factor (%)',
                            'Degradation Rate (%/yr)',
                            'Local Jobs (FTE)', 'Energy Resilience (hrs)', 'Grid Strain Reduction (%)',
                            'Community Investment (€)', 'Noise Level Impact (/10)',
                            'Payback Plausibility (/10)', 'Supply Chain Transparency (/10)',
                            'Regulatory Compliance (/10)', 'Stakeholder Reporting (/10)', 'Operational Risk (/10)'
                        ],
                        'Value': [
                            f"{int(rec.get('env_co2_reduction_tons_yr', 0)):,}",
                            f"{int(rec.get('env_land_use_sqm', 0)):,}",
                            f"{int(rec.get('env_water_savings_m3_yr', 0)):,}",
                            f"{rec.get('env_waste_factor_pct', 0.0):.1f}",
                            f"{rec.get('env_degradation_rate', 0.0) * 100:.1f}",
                            f"{rec.get('soc_local_jobs_fte', 0.0):.2f}",
                            f"{rec.get('soc_energy_resilience_hrs', 0.0):.1f}",
                            f"{rec.get('soc_grid_strain_reduction_pct', 0.0):.1f}",
                            f"€{int(rec.get('soc_community_investment_eur', 0)):,}",
                            f"{rec.get('soc_noise_level_impact_score', 0.0):.1f}",
                            f"{rec.get('gov_payback_plausibility_score', 0.0):.1f}",
                            f"{rec.get('gov_supply_chain_transparency_score', 0.0):.1f}",
                            f"{rec.get('gov_regulatory_compliance_score', 0.0):.1f}",
                            f"{rec.get('gov_stakeholder_reporting_score', 0.0):.1f}",
                            f"{rec.get('gov_operational_risk_score', 0.0):.1f}"
                        ]
                    }
                    kpi_df = pd.DataFrame(kpi_data)
                    st.table(kpi_df.set_index('Dimension'))  # Table 1 (KPIs)

                    st.subheader("ESG Category Distribution of Feasible Solutions")
                    esg_category_df = pd.DataFrame(list(intermediate['esg_category_counts'].items()),
                                                   columns=['ESG Category', 'Count'])
                    fig_esg_dist = px.bar(esg_category_df, x='ESG Category', y='Count',
                                          title='Distribution of Feasible Solutions by ESG Performance')
                    st.plotly_chart(fig_esg_dist, use_container_width=True)  # Chart 2 (ESG Category Distribution)

                with viz_tab3:
                    st.subheader("Energy Flow & Performance Metrics")

                    # Chart 3: Generation Mix Donut Chart
                    gen_mix_data = {
                        'Source': ['Solar Panels (Generated)', 'Wind Turbines (Generated)', 'Grid Import (Net)'],
                        'KWH': [
                            rec.get('solar_generation_kwh', 0),
                            rec.get('wind_generation_kwh', 0),
                            rec.get('annual_demand_kwh', 0) - (
                                        rec.get('annual_kwh_generated', 0) - rec.get('annual_kwh_exported', 0))
                        ]
                    }
                    gen_mix_df = pd.DataFrame(gen_mix_data)
                    gen_mix_df = gen_mix_df[gen_mix_df['KWH'] > 0]
                    fig_gen_mix = px.pie(gen_mix_df, values='KWH', names='Source',
                                         title='Annual Energy Source Contribution', hole=0.3)
                    st.plotly_chart(fig_gen_mix, use_container_width=True)  # Chart 3 (Generation Mix)

                    # Table 2: Detailed Performance Breakdown
                    performance_data = {
                        'Metric': ['Annual Demand (kWh)', 'Annual Generated (kWh)', 'Self-Sufficiency (%)',
                                   'Annual Exported (kWh)', 'Annual Curtailed (kWh)', 'Net Grid Import (kWh)',
                                   'Wind Contribution (%)', 'Waste Factor (%)', 'Battery Capacity (kWh)',
                                   'Energy Resilience (hours)'],
                        'Value': [
                            f"{int(rec.get('annual_demand_kwh', 0)):,}",
                            f"{int(rec.get('annual_kwh_generated', 0)):,}",
                            f"{rec.get('self_sufficiency_pct', 0.0):.1f}",
                            f"{int(rec.get('annual_kwh_exported', 0)):,}",
                            f"{int(rec.get('annual_kwh_curtailed', 0)):,}",
                            f"{int(rec.get('annual_demand_kwh', 0) - (rec.get('annual_kwh_generated', 0) - rec.get('annual_kwh_exported', 0))):,}",
                            f"{rec.get('wind_contribution_pct', 0.0):.1f}",
                            f"{rec.get('env_waste_factor_pct', 0.0):.1f}",
                            f"{int(rec.get('battery_kwh', 0)):,}",
                            f"{rec.get('soc_energy_resilience_hrs', 0.0):.1f}"
                        ]
                    }
                    performance_df = pd.DataFrame(performance_data)
                    st.table(performance_df.set_index('Metric'))  # Table 2 (Performance)

                    # Chart 4: Self-Sufficiency vs Grid Dependency Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rec.get('self_sufficiency_pct', 0.0),
                        title={'text': "Energy Self-Sufficiency"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "#0068C9"},
                               'steps': [
                                   {'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "gray"}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75,
                                             'value': rec.get('self_sufficiency_pct', 0.0)}
                               }))
                    st.plotly_chart(fig_gauge, use_container_width=True)  # Chart 4 (Self-Sufficiency Gauge)

                with viz_tab4:
                    st.subheader("Detailed Cost & Components Breakdown")
                    num_solar_panels = rec.get('num_solar_panels', 0)
                    num_wind_turbines = rec.get('num_wind_turbines', 0)
                    battery_kwh = rec.get('battery_kwh', 0)
                    total_cost = rec.get('total_cost', 0)

                    # FIX: Retrieve model constants from the API response
                    cost_per_solar_panel = model_constants.get('COST_PER_SOLAR_PANEL', 0)
                    cost_per_wind_turbine = model_constants.get('COST_PER_WIND_TURBINE', 0)
                    cost_per_battery_kwh = model_constants.get('COST_PER_BATTERY_KWH', 0)
                    installation_overhead_factor = model_constants.get('INSTALLATION_OVERHEAD_FACTOR', 0)
                    engineering_consulting_factor = model_constants.get('ENGINEERING_CONSULTING_COST_FACTOR', 0)
                    permitting_legal_factor = model_constants.get('PERMITTING_LEGAL_COST_FACTOR', 0)
                    other_components_factor = model_constants.get('OTHER_COMPONENTS_COST_FACTOR', 0)

                    annual_om_rate = model_constants.get('ANNUAL_OM_RATE', 0)
                    battery_lifetime_years = model_constants.get('BATTERY_LIFETIME_YEARS', 0)
                    project_lifetime_years = model_constants.get('PROJECT_LIFETIME_YEARS', 0)
                    financing_interest_rate = model_constants.get('FINANCING_INTEREST_RATE', 0)

                    # Recalculate hardware cost components
                    solar_hardware_cost = num_solar_panels * cost_per_solar_panel
                    wind_hardware_cost = num_wind_turbines * cost_per_wind_turbine
                    battery_hardware_cost = battery_kwh * cost_per_battery_kwh
                    initial_hardware_total = solar_hardware_cost + wind_hardware_cost + battery_hardware_cost

                    installation_cost = initial_hardware_total * installation_overhead_factor
                    engineering_consulting_cost = initial_hardware_total * engineering_consulting_factor
                    permitting_legal_cost = initial_hardware_total * permitting_legal_factor
                    other_components_cost = initial_hardware_total * other_components_factor

                    # Total Initial CAPEX (Sum of all initial costs)
                    total_initial_capex_display = initial_hardware_total + installation_cost + \
                                                  engineering_consulting_cost + permitting_legal_cost + \
                                                  other_components_cost

                    # Table 3: Component Cost Breakdown
                    cost_data = {
                        "Item": ["Solar Panels (Hardware)", "Wind Turbines (Hardware)", "Battery Storage (Hardware)",
                                 "Installation Cost (est.)", "Engineering & Consulting (est.)",
                                 "Permitting & Legal (est.)", "Other Components (est.)",
                                 "Annual O&M Cost (est.)", "Annual Battery Replacement (amortized)",
                                 "Annual Financing Cost (est.)", "Total Initial CAPEX"],
                        "Description": [f"{num_solar_panels} units @ €{cost_per_solar_panel:,}/unit",
                                        f"{num_wind_turbines} units @ €{cost_per_wind_turbine:,}/unit",
                                        f"{battery_kwh} kWh @ €{cost_per_battery_kwh:,}/kWh",
                                        f"{installation_overhead_factor * 100:.0f}% of Hardware",
                                        f"{engineering_consulting_factor * 100:.0f}% of Hardware",
                                        f"{permitting_legal_factor * 100:.0f}% of Hardware",
                                        f"{other_components_factor * 100:.0f}% of Hardware",
                                        f"{annual_om_rate * 100:.1f}% of (Hardware+Install)",
                                        f"Over {project_lifetime_years} yrs (Battery life: {battery_lifetime_years} yrs)",
                                        f"{financing_interest_rate * 100:.1f}% of Total CAPEX", ""],
                        "Cost (€)": [
                            f"€{solar_hardware_cost:,}",
                            f"€{wind_hardware_cost:,}",
                            f"€{battery_hardware_cost:,}",
                            f"€{int(installation_cost):,}",
                            f"€{int(engineering_consulting_cost):,}",
                            f"€{int(permitting_legal_cost):,}",
                            f"€{int(other_components_cost):,}",
                            f"€{int(rec.get('annual_maintenance_cost_eur', 0)):,}",
                            f"€{int(rec.get('annual_amortized_battery_replacement_cost_eur', 0)):,}",
                            f"€{int(rec.get('annual_financing_cost_eur', 0)):,}",
                            f"€{int(total_initial_capex_display):,}"
                        ]
                    }
                    cost_df = pd.DataFrame(cost_data)
                    st.table(cost_df.set_index('Item'))  # Table 3 (Cost Breakdown)

                    # Chart 5: Cost Distribution Pie Chart (Initial CAPEX only for this chart)
                    initial_capex_dist_data = {
                        'Component': ['Solar Panels (Hardware)', 'Wind Turbines (Hardware)',
                                      'Battery Storage (Hardware)', 'Installation', 'Eng. & Consult.',
                                      'Permitting & Legal', 'Other Components'],
                        'Cost': [
                            solar_hardware_cost,
                            wind_hardware_cost,
                            battery_hardware_cost,
                            installation_cost,
                            engineering_consulting_cost,
                            permitting_legal_cost,
                            other_components_cost
                        ]
                    }
                    initial_capex_dist_df = pd.DataFrame(initial_capex_dist_data)
                    initial_capex_dist_df = initial_capex_dist_df[initial_capex_dist_df['Cost'] > 0]
                    fig_initial_capex_dist = px.pie(initial_capex_dist_df, values='Cost', names='Component',
                                                    title='Initial CAPEX Distribution by Category', hole=0.3)
                    st.plotly_chart(fig_initial_capex_dist,
                                    use_container_width=True)  # Chart 5 (Initial CAPEX Distribution)

                    st.subheader("Financial Projections")
                    financial_data = {
                        'Metric': ['Initial Investment (Total CAPEX) (€)', 'Annual O&M Cost (€)',
                                   'Annual Battery Replacement (Amortized) (€)', 'Annual Financing Cost (€)',
                                   'Total Annual Savings (€)', 'Payback Period (Years)', 'Project Lifetime (Years)'],
                        'Value': [
                            f"€{int(total_initial_capex_display):,}",
                            f"€{int(rec.get('annual_maintenance_cost_eur', 0)):,}",
                            f"€{int(rec.get('annual_amortized_battery_replacement_cost_eur', 0)):,}",
                            f"€{int(rec.get('annual_financing_cost_eur', 0)):,}",
                            f"€{int(rec.get('annual_savings_eur', 0)):,}",
                            f"{rec.get('payback_period_years', 0.0):.1f}",
                            f"{project_lifetime_years}"
                        ]
                    }
                    financial_df = pd.DataFrame(financial_data)
                    st.table(financial_df.set_index('Metric'))  # Table 4 (Financial Projections)

                with viz_tab5:  # New tab for model constants
                    st.subheader("Underlying Simulation Constants")
                    st.write("These values are fixed within the simulation model for consistency.")

                    constants_to_display = {
                        "Component Costs": {
                            "COST_PER_SOLAR_PANEL (€)": model_constants.get('COST_PER_SOLAR_PANEL', 0),
                            "COST_PER_WIND_TURBINE (€)": model_constants.get('COST_PER_WIND_TURBINE', 0),
                            "COST_PER_BATTERY_KWH (€)": model_constants.get('COST_PER_BATTERY_KWH', 0),
                        },
                        "Overhead & Installation": {
                            "INSTALLATION_OVERHEAD_FACTOR (%)": model_constants.get('INSTALLATION_OVERHEAD_FACTOR',
                                                                                    0) * 100,
                            "ENGINEERING_CONSULTING_COST_FACTOR (%)": model_constants.get(
                                'ENGINEERING_CONSULTING_COST_FACTOR', 0) * 100,
                            "PERMITTING_LEGAL_COST_FACTOR (%)": model_constants.get('PERMITTING_LEGAL_COST_FACTOR',
                                                                                    0) * 100,
                            "OTHER_COMPONENTS_COST_FACTOR (%)": model_constants.get('OTHER_COMPONENTS_COST_FACTOR',
                                                                                    0) * 100,
                        },
                        "Operational & Lifetime Costs": {
                            "ANNUAL_OM_RATE (%)": model_constants.get('ANNUAL_OM_RATE', 0) * 100,
                            "BATTERY_LIFETIME_YEARS (Years)": model_constants.get('BATTERY_LIFETIME_YEARS', 0),
                            "PROJECT_LIFETIME_YEARS (Years)": model_constants.get('PROJECT_LIFETIME_YEARS', 0),
                            "FINANCING_INTEREST_RATE (%)": model_constants.get('FINANCING_INTEREST_RATE', 0) * 100,
                        }
                    }

                    for category, constants in constants_to_display.items():
                        st.markdown(f"**{category}**")
                        category_df = pd.DataFrame(list(constants.items()), columns=["Constant", "Value"]).set_index(
                            "Constant")
                        st.table(category_df)


            else:
                st.error("**No Feasible Solution Found**")
                # Improved error message for better user guidance
                st.markdown(
                    f"**Reason:** {result.get('status', 'Unknown error.').replace('AND minimum 15.0% wind contribution.', 'This is often due to very strict self-sufficiency targets, or the requirement for at least 15% wind energy, which may not be achievable with the current component combinations.')}")
                st.markdown(
                    "Try increasing the allowed grid dependency, adjusting your ESG priorities, or selecting a different building profile. Remember that a minimum of 15% wind contribution is enforced for all solutions.")

# ==============================================================================
# TAB 2: CONVERSATIONAL AI ADVISOR
# ==============================================================================
with tab2:
    st.header("Natural Language Consultation")
    st.markdown("""
        Describe your project, and our AI Advisor will use the quantitative model to find the best solution and explain it to you in a comprehensive, structured report.

        **Please note:** The underlying simulation constants (e.g., component costs, maintenance rates, financing interest, project/battery lifetimes) are **fixed within the model for consistent and auditable research results.** These are based on typical PhD-level assumptions for HRES analysis. Your natural language requests should focus on:
        *   **Building Type:** (e.g., 'hospital', 'small office')
        *   **Annual Energy Demand (kWh):** (e.g., '1.5 million kWh')
        *   **Desired Grid Dependency (%):** (e.g., 'off-grid', 'mostly self-sufficient', 'up to 40% grid reliance')
        *   **Your Priorities:** (e.g., 'low cost', 'strong environmental impact', 'social good', 'good governance', 'balanced approach')

        Examples:
        *   `"Find a solution for a small office"`
        *   `"I need a low cost solution for a small office with strong environmental impact"`
        *   `"Suggest a highly resilient system for a new hospital with a strong focus on social good"`
    """)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant",
                                           "content": "Hello! How can I help you find the optimal HRES? Please describe your project, for example: 'I need a highly resilient system for a new hospital with a strong focus on social good.'"}]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict) and "summary" in message["content"]:
                st.markdown(message["content"]["summary"])
                for detail in message["content"]["details"]:
                    with st.expander(f"**{detail['title']}**", expanded=True):
                        st.markdown(detail['content'], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("e.g., 'Help me find a low-cost solution for a small office.'"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing your request with our LLM, running quantitative simulations...▌")

            chat_result = call_api(CHAT_API_URL, {"query": prompt})

            if chat_result and chat_result.get("response"):
                ai_response = chat_result["response"]
                message_placeholder.markdown(ai_response["summary"])
                for detail in ai_response["details"]:
                    with st.expander(f"**{detail['title']}**", expanded=True):
                        st.markdown(detail['content'], unsafe_allow_html=True)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
            else:
                error_message = "Sorry, I encountered an issue processing your request. The AI chat advisor may be temporarily disabled or unavailable, or there was an error in the API. Please ensure your Azure OpenAI credentials in the .env file are valid and the API service is running correctly."
                message_placeholder.markdown(error_message)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

# ==============================================================================
# TAB 3: ML FAST PREDICTOR
# ==============================================================================
with tab3:
    st.header("⚡ ML Fast Predictor (Experimental)")
    st.markdown(
        "Use the trained Machine Learning model for quick predictions of system cost, self-sufficiency, and annual savings.")
    st.info(
        "This model is trained on the simulated dataset and offers faster, approximate predictions compared to the full MCDA pipeline.")

    # Input sliders for ML prediction
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        ml_scenario_options = {
            "🏢 Small Office": "Small_Office", "🎓 University Campus": "University_Campus",
            "🏥 Hospital": "Hospital", "🏭 Industrial Facility": "Industrial_Facility",
            "💻 Data Center": "Data_Center"
        }
        ml_selected_scenario_display = st.selectbox("Select Scenario for ML:", options=list(ml_scenario_options.keys()),
                                                    index=0, key="ml_scenario_select")
        ml_scenario_name = ml_scenario_options[ml_selected_scenario_display]

        ml_solar_panels = st.slider("Number of Solar Panels:", min_value=100, max_value=8000, value=1000, step=100,
                                    key="ml_solar_panels")
        ml_wind_turbines = st.slider("Number of Wind Turbines:", min_value=0, max_value=200, value=10, step=5,
                                     key="ml_wind_turbines")
        ml_battery_kwh = st.slider("Battery Storage (kWh):", min_value=200, max_value=10000, value=1000, step=100,
                                   key="ml_battery_kwh")

    ml_predict_button = st.button("Get ML Prediction", use_container_width=True, type="secondary")

    if ml_predict_button:
        ml_payload = {
            "scenario_name": ml_scenario_name,
            "num_solar_panels": ml_solar_panels,
            "num_wind_turbines": ml_wind_turbines,
            "battery_kwh": ml_battery_kwh
        }
        with st.spinner("Getting ML predictions..."):
            ml_result = call_api(ML_PREDICT_API_URL, ml_payload)

        if ml_result and ml_result.get("predictions"):
            predictions = ml_result["predictions"]
            st.subheader("ML Predicted Outcomes:")
            ml_pred_col1, ml_pred_col2, ml_pred_col3 = st.columns(3)
            ml_pred_col1.metric("Predicted Total Cost", f"€{int(predictions.get('total_cost', 0)):,}")
            ml_pred_col2.metric("Predicted Self-Sufficiency", f"{predictions.get('self_sufficiency_pct', 0.0):.1f}%")
            ml_pred_col3.metric("Predicted Annual Savings", f"€{int(predictions.get('annual_savings_eur', 0)):,}")
        else:
            st.error("Failed to get ML prediction.")
            if ml_result:
                st.json(ml_result)