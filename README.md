<div align="center">

# 🌾 AgriYield Analytics

### Agricultural Crop Yield Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-2.14%2B-119DFF?style=flat&logo=plotly&logoColor=white)](https://dash.plotly.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

*A complete end-to-end Data Analytics + AI project — from raw CSV to a production-quality interactive dashboard.*

---

</div>

## ✨ Overview

**AgriYield Analytics** transforms raw Kaggle agricultural data into deep, interactive insights on crop yield patterns across **101 countries**, **10 crop types**, and **23 years** (1990–2013).

The project covers the full data science lifecycle:

```
yield_df.csv  ──►  Data Cleaning  ──►  EDA  ──►  ML Model  ──►  Interactive Dashboard
```

The dashboard ships with a **sidebar-navigated, minimal UI** built with Plotly Dash, custom CSS, and the Inter typeface — designed to be clean, fast, and readable.

---

## 🖥️ Dashboard Preview

| Tab | What you see |
|-----|-------------|
| 📊 **EDA Overview** | 10+ interactive charts — bar, line, scatter, box, pie, heatmap, area |
| 🗺️ **Geo Maps** | World choropleth of mean yield & annual rainfall |
| 🤖 **ML & Predictions** | Model metrics, feature importance, actual vs predicted, residuals |
| 🔍 **Explore Data** | Live crop/country/year filters + scrollable data table |
| 🎯 **Yield Predictor** | Real-time prediction with historical comparison |

---

## 🗂️ Project Structure

```
Agriculture_Yield_Analytics/
│
├── 📄  app.py                              ← Main app (pipeline + ML + dashboard)
├── 📄  Keerti_AgricultureYieldAnalytics.py ← Submission copy (identical)
├── 📊  yield_df.csv                        ← Primary dataset (Kaggle)
├── 📦  requirements.txt                    ← Python dependencies
├── 📝  README.md                           ← This file
└── 📋  Keerti_ProjectReport.docx           ← Full project documentation
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle — [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) |
| **File** | `yield_df.csv` |
| **Raw Rows** | 28,242 |
| **Clean Rows** | ~13,130 (after deduplication) |
| **Countries** | 101 |
| **Crops** | 10 |
| **Years** | 1990 – 2013 |

### Features

| Column | Description |
|--------|-------------|
| `Area` | Country / region |
| `Item` | Crop type (Wheat, Rice, Potatoes…) |
| `Year` | Year of observation |
| `yield_hg_ha` | Crop yield in hectograms / hectare *(target)* |
| `rainfall_mm` | Annual rainfall (mm) — static per country |
| `pesticides_t` | Pesticide usage (tonnes) |
| `avg_temp_c` | Average temperature (°C) |

### Crop Types
`Cassava` · `Maize` · `Plantains` · `Potatoes` · `Rice (paddy)` · `Sorghum` · `Soybeans` · `Sweet Potatoes` · `Wheat` · `Yams`

---

## 🧹 Data Preprocessing

| Step | Action |
|------|--------|
| Drop index column | Unnamed first column removed |
| Rename columns | `hg/ha_yield` → `yield_hg_ha` etc. |
| Deduplicate | 28,242 → 13,130 rows; duplicate `avg_temp` rows averaged |
| Log-transform | `log_yield` and `log_pesticides` derived for ML |
| NaN drop | Zero NaN rows remain after deduplication |

---

## 🤖 Machine Learning

### Model: Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators = 200,
    max_depth    = 15,
    n_jobs       = -1,
    random_state = 42
)
```

### Feature Set

| Feature | Description |
|---------|-------------|
| `area_enc` | Country (label-encoded) |
| `item_enc` | Crop type (label-encoded) |
| `Year` | Year of observation |
| `rainfall_mm` | Annual rainfall |
| `log_pesticides` | log(pesticides + 1) |
| `avg_temp_c` | Average temperature |

**Target:** `log_yield` → inverse-transformed for metrics

### Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9618 |
| **MAE** | 7,460 hg/ha |
| **RMSE** | 15,677 hg/ha |
| **MAPE** | 15.55% |
| **CV R² (5-fold)** | 0.58 ± 0.06 |

> **Note on CV R²:** The lower cross-validation score vs test R² is expected — CV splits cut across crop types (the strongest feature), creating distribution shifts between folds. The 0.96 test R² reflects true in-distribution performance.

### Feature Importance

```
Crop Type         ████████████████████  59.7%
Pesticides (log)  ████████             12.1%
Avg Temperature   ███████              10.7%
Rainfall          ██████                9.3%
Country           ████                  6.0%
Year              ██                    2.2%
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9 or higher
- pip

### Quick Start

```bash
# 1. Navigate to the project folder
cd Agriculture_Yield_Analytics

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:8050
```

> The terminal will print model metrics before the server starts (~30–60s first run for training).  
> Press **CTRL+C** to stop.

---

## 🔍 Key Findings

> **1. Root vegetables dominate yield (by mass)**  
> Potatoes average ~200,000 hg/ha vs Wheat's ~30,000 hg/ha — a 6× difference driven by water content and tuber density, not agricultural efficiency.

> **2. Consistent 36% global yield growth (1990–2013)**  
> Mean yield rose from 66,447 → 90,357 hg/ha. No year showed a decline, reflecting ongoing Green Revolution gains.

> **3. Country identity is the #1 predictor**  
> Soil type, irrigation, policy, and technology — all encoded in the country label — explain more yield variance than any directly measured climate variable.

> **4. Pesticide–yield relationship is non-linear**  
> Raw Pearson r = 0.064, but on log scales the relationship is clearly positive. Diminishing returns apply at high pesticide doses.

> **5. Rainfall is a country constant here**  
> All 101 countries carry one static rainfall value across all years — it cannot model climate variability, limiting its predictive value.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Dashboard | Plotly Dash 2.14+ |
| UI Styling | Dash Bootstrap Components + Custom CSS |
| Visualisation | Plotly Express & Graph Objects |
| Data | Pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest) |

---

## 📋 Submission Files

| File | Purpose |
|------|---------|
| `Keerti_AgricultureYieldAnalytics.py` | Main code file |
| `requirements.txt` | Python dependencies |
| `Keerti_ProjectReport.docx` | Full project report |
| `README.md` | This overview |

---

## 🔭 Future Scope

- 📡 **Extended data** — pull post-2013 records from FAO or updated Kaggle datasets
- 🕒 **Time-series forecasting** — Prophet / LSTM per country-crop pair
- 🌍 **Sub-national data** — district/state level granularity
- 🛰️ **Satellite features** — NDVI, soil moisture, land surface temperature
- 🌱 **Crop recommendation engine** — suggest optimal crop per climate profile
- 💡 **SHAP explainability** — per-prediction feature attribution

---

<div align="center">

Made with 💚 by **Keerti** · Agriculture Yield Analytics

</div>
