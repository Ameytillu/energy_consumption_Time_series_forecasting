# ⚡ Energy Demand Forecasting System

> An end-to-end machine learning pipeline for predicting hourly electricity demand in a smart-grid environment — from raw multi-source Excel data to a live interactive Streamlit application.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling & Training](#modeling--training)
- [Results & Findings](#results--findings)
- [Streamlit App](#streamlit-app)
- [Installation & Setup](#installation--setup)
- [Requirements](#requirements)

---

## Project Overview

Reliable electricity demand forecasting is critical for grid stability, renewable energy integration, and operational cost reduction. This project builds a production-ready forecasting pipeline that ingests multi-building smart-grid sensor data and predicts **total hourly electric load (kW)** using machine learning.

The pipeline covers the full ML lifecycle:
- Multi-source Excel data merging and automated cleaning
- KNN-based missing value imputation
- 30+ engineered temporal, weather, and lag features
- Leakage-safe time-ordered train/test splitting
- Benchmarking of 6 models from linear baselines to gradient boosting and ensembles
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Deployment via an interactive Streamlit dashboard

---

## Project Structure

```
├── Data_Output/                  # Cleaned and feature-engineered CSVs
├── Project_data/                 # Raw Excel files (multi-building, multi-sensor)
├── models/                       # Serialized trained model and scaler
├── notebooks/
│   ├── data_cleaning.ipynb       # EDA, imputation, outlier capping
│   ├── energy_consumption_ml.ipynb  # Full modeling and evaluation
│   └── final_modeling.ipynb      # Final leakage-free pipeline
├── main.py                       # Entry point
├── merging_data.py               # Multi-file Excel merge logic
├── refining_data.py              # Column dropping and renaming
├── streamlit_app.py              # Interactive forecasting dashboard
├── test_app.py                   # App unit tests
├── test_predictions.py           # Prediction validation tests
└── requirements.txt
```

---

## Dataset

The raw dataset consists of hourly sensor readings from a smart-grid facility spanning multiple buildings and weather stations, stored across multiple Excel files grouped by data category.

**Key columns after cleaning:**

| Column | Description |
|---|---|
| `Outdoor_Air_Temp_C` | Outdoor temperature in Celsius |
| `Outdoor_Air_Humidity_percent` | Relative humidity (%) |
| `Solar_Irradiation_W` | Horizontal solar irradiation (W/m²) |
| `Wind_Speed_ms` | Wind speed (m/s) |
| `Electric_Load_A_kW` | Electrical load — Building A |
| `Electric_Load_B_kW` | Electrical load — Building B |
| `Cooling_Load_A_kW` | Cooling load — Building A |
| `Heating_Load_A_kW` | Heating load — Building A |
| `PV_Generation_kW` | Solar PV generation output |
| `Grid_Import_kW` | Electricity imported from grid |
| `TARGET_Total_Electric_Load_kW` | **Target variable** (A + B combined load) |

---

## Data Cleaning & Preprocessing

Data cleaning is handled across `merging_data.py`, `refining_data.py`, and `data_cleaning.ipynb`.

### 1. Multi-File Merging
Raw data is distributed across grouped subfolders. The `merge_paths()` function scans the `Project_data/` directory, collects all `.xlsx` paths per group, and `data_merger()` concatenates each group and performs a date-indexed outer join across all groups — producing a single unified DataFrame.

```python
# Key logic
master_group_df = pd.concat(group_dfs_list, ignore_index=True)
master_group_df[DATE_COLUMN_NAME] = pd.to_datetime(master_group_df[DATE_COLUMN_NAME], errors='coerce')
final_combined_df = pd.concat(master_dfs_to_join, axis=1, join='outer')
```

### 2. Column Dropping & Renaming
Over 50 raw columns were reduced to the most informative ones. Sparse columns (`Unnamed: 5–8`), GJ-unit duplicates of kW columns, and ambiguous gas consumption fields were dropped. Remaining columns were renamed for clarity (e.g., `'Horizontal solar irradition (W)'` → `'Solar_Irradiation_W'`).

### 3. Target Variable Construction & Outlier Capping
The target is the sum of both building loads. Extreme outliers were capped using a 3×IQR rule on the 1st–99th percentile range to prevent skewed training.

```python
imputed_data['TARGET_Total_Electric_Load_kW'] = (
    imputed_data['Electric_Load_A_kW'] + imputed_data['Electric_Load_B_kW']
)
imputed_data['TARGET_Total_Electric_Load_kW'] = imputed_data[
    'TARGET_Total_Electric_Load_kW'
].clip(lower=lower_bound, upper=upper_bound)
```

### 4. KNN Imputation
Rather than simple mean/median fill, `KNNImputer(n_neighbors=5)` was applied to all numerical columns. This respects the local neighborhood of each missing value, producing more temporally coherent gap-fills — particularly during multi-hour sensor outages.

```python
imputer = KNNImputer(n_neighbors=5)
imputed_data[numerical_cols] = imputer.fit_transform(df[numerical_cols])
```

### 5. Data Leakage Removal
Initial experiments yielded near-perfect R² ≈ 1.0 because generation and component-level columns were included as features. All 14 leakage columns were identified and dropped before modeling:

```python
LEAKAGE_COLS = [
    'PV_Generation_kW', 'Grid_Import_kW', 'GasEngine_Generation_kW',
    'FuelCell_Generation_kW', 'Electric_Load_A_kW', 'Electric_Load_B_kW',
    'Cooling_Load_A_kW', 'Heating_Load_A_kW', 'HotWater_Load_A_kW',
    'Gas1_Input_GJh', 'Gas2_Input_GJh', 'FuelCell1_Input_GJh',
    'FuelCell2_Input_GJh', 'FuelCell3_Input_GJh'
]
```

---

## Feature Engineering

30+ features were engineered from the leakage-free dataset, grouped into five categories:

### Time Features
Extracted from the datetime index to capture human activity cycles and calendar effects:
`year`, `month`, `day`, `hour`, `weekday`, `dayofyear`, `quarter`, `week_of_year`, `season`, `is_weekend`, `is_peak_hour` (17:00–20:00), `Is_Holiday` (US public holidays via the `holidays` library).

### Cyclical Encodings
Raw integer time values create artificial discontinuities (e.g., hour 23 → 0 treated as a large jump). Sine/cosine encoding preserves circular continuity:

```python
dfe['Hour_sin']       = np.sin(2 * np.pi * dfe['hour'] / 24)
dfe['Hour_cos']       = np.cos(2 * np.pi * dfe['hour'] / 24)
dfe['DayOfWeek_sin']  = np.sin(2 * np.pi * dfe['weekday'] / 7)
dfe['DayOfWeek_cos']  = np.cos(2 * np.pi * dfe['weekday'] / 7)
```

### Weather-Derived Features
Temperature effects on HVAC demand are captured via:
- **HDD** (Heating Degree Days): `max(0, 18 - Temp)`
- **CDD** (Cooling Degree Days): `max(0, Temp - 18)`
- **Temp_x_Hour**: interaction term for temperature stress by hour
- **Is_High_Temp** / **Is_Low_Temp**: binary flags for extreme conditions
- **Comfort_Index**: composite metric combining temperature, humidity, and wind

### Lag Features (Past-Only, Shift-Safe)
Lag features capture autocorrelation in energy demand without introducing future data leakage. All lags are shifted by at least 6 hours:

```python
dfe['Lag_6h_Load']  = dfe[TARGET].shift(6)
dfe['Lag_12h_Load'] = dfe[TARGET].shift(12)
dfe['Lag_24h_Load'] = dfe[TARGET].shift(24)
dfe['Lag_48h_Load'] = dfe[TARGET].shift(48)
```

### Rolling & Smoothed Features
Capture local trends and volatility. All rolling operations are shifted by 1 before computing to avoid same-timestep leakage:

```python
dfe['Rolling_Mean_12h'] = dfe[TARGET].shift(1).rolling(12).mean()
dfe['Rolling_Mean_24h'] = dfe[TARGET].shift(1).rolling(24).mean()
dfe['Rolling_Std_24h']  = dfe[TARGET].shift(1).rolling(24).std()
dfe['EWMA_24h_Load']    = dfe[TARGET].shift(1).ewm(span=24, adjust=False).mean()
```

---

## Exploratory Data Analysis

Key EDA findings before modeling:

- **Daily seasonality**: Load is lowest from 1–6 AM, rises sharply after 7 AM, peaks between 11 AM and 3 PM, and gradually declines through the evening.
- **Right-skewed distribution**: Most energy values cluster between 900–1,500 kW, with infrequent spikes above 2,500 kW caused by weather extremes.
- **Strong daily autocorrelation**: Lag_24h and Lag_48h show the highest correlation to the target, confirming that the same-hour reading from the previous day is the strongest standalone predictor.
- **Temperature influence**: CDD, HDD, and Temp_x_Hour show moderate correlation, reflecting HVAC-driven demand changes.
- **Yearly patterns**: Daily-sampled load plots show consistent seasonal cycles across all years in the dataset.

---

## Modeling & Training

### Train / Test Split
An **80/20 time-ordered split** was used — no shuffling — to respect the temporal structure of the data and prevent future data from leaking into training.

```python
split_idx = int(len(dfe) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

### Models Evaluated

**1. Baseline — 24-Hour Lag**
Predicts the load at the same hour from the previous day. Provides a realistic benchmark reflecting daily periodicity.

**2. Linear Regression**
Included to establish a foundational performance floor. Underfits due to the nonlinear nature of smart-grid demand.
- MAE: 144.26 kW | RMSE: 187.07 kW | R²: 0.8783

**3. Ridge Regression**
L2-regularized linear model to reduce overfitting on correlated features. Marginal improvement over plain linear regression.
- MAE: 144.25 kW | RMSE: 187.07 kW | R²: 0.8783

**4. HistGradientBoostingRegressor (Default)**
Gradient boosted decision trees with native missing-value support. Substantial jump over linear models.
- MAE: 69.58 kW | RMSE: 100.91 kW | R²: 0.9646

**5. XGBoost**
Slightly edges out default HGB on accuracy.
- MAE: 66.42 kW | RMSE: 95.45 kW | R²: 0.9683

**6. HGB + RandomizedSearchCV (Best Model)**
Broad hyperparameter search over learning rate, max depth, regularization, subsample ratio, and number of iterations. Produces the most stable forecasts across all hours.
- Error range: **~20–60 kW** | Best overall performance

**7. Stacking Ensemble**
Combines Ridge + HGB as base learners. Competitive overall but struggles at high-load peak hours, occasionally producing large negative errors (−150 to −200 kW).

**8. LSTM Neural Network**
Deep learning sequence model for time-series context.
- MAE: 88.23 kW | RMSE: 121.12 kW | R²: 0.9490 | MAPE: 6.48%

### Hyperparameter Tuning
`RandomizedSearchCV` explored combinations of:
```python
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, None],
    'l2_regularization': [0.0, 0.1, 1.0],
    'subsample': [0.7, 0.8, 1.0],
    'max_iter': [200, 400, 600]
}
```

---

## Results & Findings

### Model Performance Summary

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Baseline (24h Lag) | — | — | Benchmark |
| Linear Regression | 144.26 | 187.07 | 0.8783 |
| Ridge Regression | 144.25 | 187.07 | 0.8783 |
| HGB Default | 69.58 | 100.91 | 0.9646 |
| XGBoost | 66.42 | 95.45 | 0.9683 |
| LSTM | 88.23 | 121.12 | 0.9490 |
| **HGB + RandomizedSearchCV** | **~20–60** | **Lowest** | **Best** |
| Stacking Ensemble | 40–200 | Variable | Good overall |

### Key Findings

1. **Data leakage was the most critical issue to fix.** Including generation and component columns (Grid_Import_kW, Electric_Load_A/B_kW, etc.) inflated R² to ~1.0. Removing all 14 leakage columns and rebuilding the feature set was the highest-impact correctness step in the project.

2. **Lag features are the strongest predictors.** Lag_24h_Load, Lag_48h_Load, EWMA_24h_Load, and Rolling_Mean_24h consistently rank as the top correlated features to the target — reflecting the strong daily autocorrelation in electricity demand.

3. **Cyclical encoding outperforms raw time integers.** Encoding hour and weekday as sine/cosine pairs preserved circular continuity (e.g., 23:00 → 00:00 is not a large jump), measurably improving forecast accuracy during overnight and weekend transitions.

4. **Gradient boosting achieves a ~5× error reduction over linear models.** Linear and Ridge regression produced errors of 100–300 kW due to their inability to capture nonlinear patterns. The tuned HGB model reduced this to 20–60 kW.

5. **Temperature is the primary weather driver.** HDD, CDD, and Comfort_Index capture HVAC-driven demand spikes during extreme cold and heat events. These peak-hour extremes remain the hardest to predict accurately across all models.

6. **Clear daily consumption cycle.** EDA confirmed load is consistently lowest from 1–6 AM, peaks between 11 AM and 3 PM, and declines through the evening — a pattern the lag and cyclic features encode explicitly.

7. **Stacking did not surpass tuned boosting.** Despite combining multiple models, the stacking ensemble underperformed the tuned HGB at high-demand peaks, indicating that the weaknesses of the base learners (particularly Ridge) compounded rather than cancelled.

---

## Streamlit App

The trained model and scaler are serialized and loaded at runtime by a Streamlit dashboard (`streamlit_app.py`), allowing users to interactively forecast energy demand without any code.

### Features
- **Time Settings**: Adjust the hour of day via slider
- **Weather Conditions**: Set outdoor temperature (°C), humidity (%), solar irradiation (W/m²), and wind speed (m/s)
- **Load Adjustments**: Configure Electric Load A & B, Cooling Load A, and Heating Load A
- **Generation Inputs**: PV generation (kW) and grid import (kW)
- **Current Settings Panel**: Real-time summary of all selected parameters
- **Predict Button**: Runs the loaded ML model and returns the forecasted total electric demand in kW

### Running the App

```bash
streamlit run streamlit_app.py
```

The app loads the model from the `models/` directory. Ensure the trained model file and scaler are present before launching.

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/energy-consumption-forecasting.git
cd energy-consumption-forecasting

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data in Project_data/ folder

# 5. Run the data pipeline
python main.py

# 6. Launch the Streamlit app
streamlit run streamlit_app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
streamlit
holidays
openpyxl
joblib
tensorflow        # for LSTM model
```

Full pinned versions available in `requirements.txt`.

---

