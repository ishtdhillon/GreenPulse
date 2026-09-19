# 🌱 GreenPulse

### AI-Powered Energy Intelligence for Sustainable Campuses

GreenPulse is an AI-powered energy intelligence prototype designed to help educational institutions understand energy consumption, identify unusual usage patterns, and support data-driven sustainability decisions.

The system combines **machine learning, anomaly detection, feature engineering, and interactive visualization** into a single dashboard.

> **Prototype note:** GreenPulse is demonstrated using the UCI Appliances Energy Prediction public dataset, which represents building-level energy consumption rather than actual campus meter data. Real campus deployment would require campus-specific energy data and validation.

---

## 🎯 Problem Statement

Educational institutions consume significant amounts of electricity across lighting, HVAC systems, equipment, and other infrastructure.

Traditional energy monitoring can show how much electricity was consumed, but it may not clearly answer:

* What consumption should be expected?
* When is usage unusually high or low?
* Which periods deserve investigation?
* How can historical patterns support better energy decisions?

GreenPulse addresses this gap by combining **energy prediction and anomaly detection** with an interactive sustainability dashboard.

---

## 🌍 SDG Alignment

### SDG 7 — Affordable and Clean Energy

GreenPulse supports more efficient energy use by providing data-driven insights into consumption patterns and potential areas of energy wastage.

### SDG 13 — Climate Action

Improved energy awareness and efficiency can support efforts to reduce unnecessary energy consumption and associated emissions.

---

## 💡 Solution Overview

GreenPulse follows this workflow:

```text
Building Energy Data
        ↓
Data Cleaning & Feature Engineering
        ↓
Energy Consumption Prediction
        ↓
Anomaly Detection
        ↓
Pattern Analysis
        ↓
Sustainability Insights
        ↓
Interactive Dashboard
```

The system does not automatically change operational settings. Instead, it provides signals and recommendations that can be reviewed by humans before action is taken.

---

## 🤖 AI & Machine Learning

### 1. Energy Consumption Prediction

A **HistGradientBoostingRegressor** is used to estimate energy consumption based on:

* Time of day
* Day of week
* Weekend/weekday patterns
* Temperature
* Humidity
* Weather conditions
* Lighting consumption
* Previous energy consumption
* Rolling consumption averages

### 2. Anomaly Detection

An **Isolation Forest** identifies unusual consumption patterns that differ from typical observations.

An anomaly does not automatically indicate equipment failure or energy wastage. It represents a pattern that may require further investigation.

### 3. Feature Engineering

The project uses:

* Lag features
* Rolling averages
* Cyclic time features
* Environmental variables
* Historical energy patterns

---

## 📊 Model Performance

The prediction model was evaluated on a held-out portion of the dataset.

| Metric |       Result |
| ------ | -----------: |
| MAE    | **31.39 Wh** |
| RMSE   | **60.36 Wh** |
| R²     |   **0.5277** |

These results represent the prototype's performance on the public demonstration dataset and should not be interpreted as performance on real campus energy data.

---

## 🖥️ Dashboard Features

### 📊 Energy Overview

Provides:

* Total energy consumption
* Predicted consumption
* Estimated electricity cost
* Estimated CO₂ emissions
* Number of detected anomalies

### 📈 Energy Trends

Visualizes:

* Actual energy consumption
* Predicted consumption
* Average consumption
* Peak usage hour

### 🚨 Anomaly Monitor

Displays unusual consumption intervals detected by the Isolation Forest model.

### 🤖 AI Sustainability Advisor

Provides data-driven observations and suggested areas for investigation, such as:

* Peak-hour loads
* Equipment schedules
* Lighting usage
* After-hours operation

### 🛡️ Responsible AI

The dashboard highlights:

* Prediction uncertainty
* Human oversight
* Data privacy
* Fairness considerations
* Deployment limitations

---

## 🧰 Technology Stack

**Programming**

* Python

**Machine Learning**

* Scikit-learn
* HistGradientBoostingRegressor
* Isolation Forest

**Data Processing**

* Pandas
* NumPy

**Model Storage**

* Joblib

**Dashboard**

* Streamlit

**Dataset**

* UCI Appliances Energy Prediction Dataset

---

## 📂 Project Structure

```text
GreenPulse/
│
├── app.py
├── train_model.py
│
├── energy_model.pkl
├── anomaly_model.pkl
├── model_features.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/GreenPulse-AI-Energy-Intelligence.git
```

Move into the project directory:

```bash
cd GreenPulse-AI-Energy-Intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

First, train the models if the model files are not already available:

```bash
python train_model.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

The application will open locally in your browser.

---

## 📚 Dataset

GreenPulse uses the **UCI Appliances Energy Prediction Dataset** for prototype demonstration.

The dataset contains building energy measurements together with environmental and weather-related variables.

The dataset is used only as a public demonstration source. It does **not** represent actual energy consumption from a university campus.

For a real deployment, the system would need to be retrained and validated using campus-specific meter data.

---

## 🛡️ Responsible AI

GreenPulse follows several responsible AI principles:

### Transparency

The system clearly distinguishes predicted values from observed consumption.

### Human Oversight

Anomaly alerts are intended to support investigation rather than automatically trigger operational decisions.

### Privacy

The prototype does not require personal identifiers.

### Fairness

Before real deployment, performance should be evaluated across different buildings, seasons, occupancy patterns, and operating conditions.

### Data Governance

Real-world deployment should follow appropriate institutional data-management and security practices.

---

## ⚠️ Limitations

The current prototype has several limitations:

* It uses public building-energy data rather than actual campus data.
* The model is demonstrated on one dataset and requires validation before deployment.
* Anomalies do not identify their underlying cause.
* Electricity cost and CO₂ conversion values in the dashboard are configurable prototype assumptions.
* Real campus deployment would require continuous meter data and model recalibration.

---

## 🔮 Future Scope

Future versions of GreenPulse could include:

* Real-time campus smart-meter integration
* Building-level energy comparison
* Weather API integration
* Occupancy-aware prediction
* Automated sustainability reports
* Energy-saving opportunity estimation
* Campus-specific model retraining
* Historical sustainability benchmarking

---

## 🌱 Expected Impact

GreenPulse aims to transform raw energy measurements into understandable insights that can help institu
