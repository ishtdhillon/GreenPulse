import streamlit as st
import pandas as pd
import numpy as np
import joblib

from ucimlrepo import fetch_ucirepo


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GreenPulse",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 42px !important;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("🌱 GreenPulse")

st.subheader(
    "AI-Powered Energy Intelligence for Sustainable Campuses"
)

st.write(
    "GreenPulse uses machine learning to forecast energy consumption, "
    "identify unusual usage patterns and generate actionable "
    "sustainability insights."
)

st.info(
    "Prototype note: GreenPulse is demonstrated using a public "
    "building-energy dataset. It is not actual campus meter data. "
    "Real deployment would require campus-specific energy data."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    dataset = fetch_ucirepo(id=374)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    # Fix datetime format
    df["date"] = df["date"].astype(str).str.replace(
        r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})",
        r"\1 \2",
        regex=True
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


df = load_data()


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(data):

    data = data.copy()

    # Time features
    data["hour"] = (
        data["date"].dt.hour
        + data["date"].dt.minute / 60
    )

    data["day_of_week"] = data["date"].dt.dayofweek

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["month"] = data["date"].dt.month

    # Cyclic time features
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # Consumption history
    data["lag_1"] = data["Appliances"].shift(1)
    data["lag_2"] = data["Appliances"].shift(2)
    data["lag_3"] = data["Appliances"].shift(3)

    data["lag_6"] = data["Appliances"].shift(6)

    data["lag_144"] = data["Appliances"].shift(144)

    # Rolling averages
    data["rolling_mean_3"] = (
        data["Appliances"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    data["rolling_mean_6"] = (
        data["Appliances"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    data["rolling_mean_12"] = (
        data["Appliances"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    return data


df = create_features(df)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    energy_model = joblib.load(
        "energy_model.pkl"
    )

    anomaly_model = joblib.load(
        "anomaly_model.pkl"
    )

    model_features = joblib.load(
        "model_features.pkl"
    )

    return (
        energy_model,
        anomaly_model,
        model_features
    )


energy_model, anomaly_model, model_features = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Dashboard Controls")

min_date = df["date"].dt.date.min()
max_date = df["date"].dt.date.max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

st.sidebar.divider()

st.sidebar.subheader("Prototype Assumptions")

electricity_rate = st.sidebar.number_input(
    "Electricity rate (₹/kWh)",
    min_value=0.0,
    value=8.0,
    step=0.5
)

co2_factor = st.sidebar.number_input(
    "CO₂ factor (kg/kWh)",
    min_value=0.0,
    value=0.82,
    step=0.01
)

st.sidebar.caption(
    "These values are illustrative prototype assumptions. "
    "Replace them with local campus values for real deployment."
)


# =========================================================
# DATE FILTER
# =========================================================

if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date = date_range[0]
    end_date = date_range[1]

else:

    start_date = date_range
    end_date = date_range


filtered = df[
    (df["date"].dt.date >= start_date)
    &
    (df["date"].dt.date <= end_date)
].copy()


# =========================================================
# PREDICTION
# =========================================================

filtered = filtered.dropna(
    subset=model_features
).copy()


filtered["Predicted_Wh"] = energy_model.predict(
    filtered[model_features]
)

filtered["Actual_kWh"] = (
    filtered["Appliances"] / 1000
)

filtered["Predicted_kWh"] = (
    filtered["Predicted_Wh"] / 1000
)


# =========================================================
# ANOMALY DETECTION
# =========================================================

anomaly_features = pd.DataFrame({

    "Appliances":
        filtered["Appliances"],

    "lights":
        filtered["lights"],

    "hour_num":
        filtered["hour"],

    "day_of_week":
        filtered["day_of_week"],

    "T_out":
        filtered["T_out"],

    "RH_out":
        filtered["RH_out"]

})


filtered["anomaly"] = anomaly_model.predict(
    anomaly_features
)

filtered["is_anomaly"] = (
    filtered["anomaly"] == -1
)


# =========================================================
# KEY METRICS
# =========================================================

total_energy = filtered["Actual_kWh"].sum()

predicted_energy = (
    filtered["Predicted_kWh"].sum()
)

average_energy = (
    filtered["Actual_kWh"].mean()
)

peak_hour = int(
    filtered
    .groupby(filtered["date"].dt.hour)["Actual_kWh"]
    .mean()
    .idxmax()
)

anomaly_count = int(
    filtered["is_anomaly"].sum()
)

estimated_cost = (
    total_energy * electricity_rate
)

estimated_co2 = (
    total_energy * co2_factor
)


# =========================================================
# OVERVIEW
# =========================================================

st.markdown("## 📊 Energy Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Energy Consumption",
    f"{total_energy:,.1f} kWh"
)

col2.metric(
    "Predicted Consumption",
    f"{predicted_energy:,.1f} kWh"
)

col3.metric(
    "Estimated Cost",
    f"₹{estimated_cost:,.0f}"
)

col4.metric(
    "Estimated CO₂",
    f"{estimated_co2:,.1f} kg"
)

col5.metric(
    "Anomalies",
    anomaly_count
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Energy Trends",
        "🚨 Anomaly Monitor",
        "🤖 AI Sustainability Advisor",
        "🛡️ Responsible AI"
    ]
)


# =========================================================
# TAB 1 — ENERGY TRENDS
# =========================================================

with tab1:

    st.header("Energy Consumption Trends")

    chart_data = (
        filtered[
            [
                "date",
                "Actual_kWh",
                "Predicted_kWh"
            ]
        ]
        .set_index("date")
    )

    st.line_chart(
        chart_data,
        height=400
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Consumption",
        f"{average_energy:.3f} kWh"
    )

    col2.metric(
        "Peak Usage Hour",
        f"{peak_hour:02d}:00"
    )

    difference = (
        predicted_energy - total_energy
    )

    col3.metric(
        "Actual vs Predicted",
        f"{difference:+,.1f} kWh"
    )

    st.markdown(
        "### 🔎 What the graph shows"
    )

    st.write(
        "The chart compares observed energy consumption "
        "with the model's predicted consumption. Differences "
        "between the two can help identify periods that "
        "deserve further investigation."
    )


# =========================================================
# TAB 2 — ANOMALY MONITOR
# =========================================================

with tab2:

    st.header("🚨 Anomaly Monitor")

    anomalies = filtered[
        filtered["is_anomaly"]
    ].copy()

    if len(anomalies) == 0:

        st.success(
            "No unusual consumption patterns were detected "
            "for the selected period."
        )

    else:

        st.warning(
            f"{len(anomalies)} unusual intervals detected."
        )

        display_columns = [
            "date",
            "Appliances",
            "Predicted_Wh",
            "T_out",
            "RH_out"
        ]

        st.dataframe(
            anomalies[
                display_columns
            ]
            .sort_values(
                "date",
                ascending=False
            ),
            use_container_width=True
        )

        st.caption(
            "An anomaly indicates an unusual pattern "
            "detected by the model. It does not prove "
            "equipment failure, human error or energy wastage."
        )


# =========================================================
# TAB 3 — AI SUSTAINABILITY ADVISOR
# =========================================================

with tab3:

    st.header("🤖 AI Sustainability Advisor")

    actual = total_energy
    predicted = predicted_energy

    if predicted > 0:

        deviation = (
            (actual - predicted)
            / predicted
        ) * 100

    else:

        deviation = 0


    st.markdown("### 📌 Current Insight")

    if deviation > 8:

        st.warning(
            f"Observed consumption is approximately "
            f"{deviation:.1f}% above the model's expected level."
        )

        st.write(
            "Possible areas for investigation include "
            "peak-hour loads, equipment schedules, lighting "
            "usage and after-hours operation."
        )

    elif deviation < -8:

        st.success(
            f"Observed consumption is approximately "
            f"{abs(deviation):.1f}% below the model's expected level."
        )

        st.write(
            "Review the operating practices during this period "
            "to identify patterns that could potentially be "
            "replicated."
        )

    else:

        st.info(
            "Observed consumption is broadly aligned with "
            "the model's expected pattern."
        )


    st.markdown("### 💡 Recommended Actions")

    recommendations = [
        "Investigate intervals with unusually high consumption.",
        "Review equipment operating schedules during peak hours.",
        "Check whether lighting and HVAC systems remain active during low-occupancy periods.",
        "Compare repeated anomalies across different days before taking action.",
        "Validate model alerts against actual meter readings."
    ]

    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.write(
            f"**{i}.** {recommendation}"
        )


    st.markdown("### 🌱 Sustainability Focus")

    st.write(
        "The purpose of GreenPulse is not to automatically "
        "make operational decisions. Instead, it provides "
        "data-driven signals that can help facility teams "
        "identify where energy efficiency efforts may be "
        "worth investigating."
    )


# =========================================================
# TAB 4 — RESPONSIBLE AI
# =========================================================

with tab4:

    st.header("🛡️ Responsible AI Considerations")

    st.markdown("""
    **Transparency**  
    Predictions are estimates generated from historical
    building-energy patterns and should not be treated as
    guaranteed future values.

    **Human Oversight**  
    Anomaly alerts identify unusual patterns but do not
    determine their cause. Human validation is required
    before operational action.

    **Data Privacy**  
    The prototype does not require personal identifiers.
    A real campus implementation should follow appropriate
    data-governance and privacy practices.

    **Fairness**  
    Model performance should be evaluated across different
    buildings, seasons and operating conditions before
    deployment.

    **Deployment Limitation**  
    The current demonstration uses public building-energy
    data rather than actual campus meter data. A real
    deployment would require campus-specific data,
    validation and recalibration.

    **Responsible Recommendations**  
    GreenPulse provides recommendations for investigation,
    not automatic instructions to change equipment or
    operational settings.
    """)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

with st.expander("🔬 Model Information"):

    st.write(
        "**Prediction Model:** "
        "HistGradientBoostingRegressor"
    )

    st.write(
        "**Anomaly Detection:** "
        "Isolation Forest"
    )

    st.write(
        "**Prediction Features:** "
        f"{len(model_features)}"
    )

    st.write(
        "**Training Dataset:** "
        "UCI Appliances Energy Prediction"
    )

    st.write(
        "**Test R²:** 0.5277"
    )

    st.write(
        "**Test MAE:** 31.39 Wh"
    )

    st.write(
        "**Test RMSE:** 60.36 Wh"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "GreenPulse | 1M1B AI for Sustainability | "
    "SDG 7: Affordable and Clean Energy | "
    "SDG 13: Climate Action"
)