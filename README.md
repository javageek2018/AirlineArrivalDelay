# Predicting U.S. Domestic Flight Arrival Delays Under Distributional Shift

A machine learning project that predicts U.S. domestic flight arrival delays (≥ 15 minutes) using BTS on-time performance data and NOAA weather observations, with a focus on model robustness under the COVID-19 distributional shift.

---

## Research Question

Which machine learning models remain robust at predicting U.S. domestic flight arrival delays when trained on pre- and during-pandemic data and tested on the post-COVID operational environment?

---

## Dataset

| Source | Details |
|--------|---------|
| **BTS On-Time Performance** | ~45M domestic flights, 2018–2024 |
| **Iowa Mesonet ASOS Weather** | Hourly weather at origin & destination airports, joined at T-2 hours before scheduled departure |

**Target:** `ArrDel15` = 1 when arrival is ≥ 15 minutes late (~18.3% positive rate)

**Temporal Split:**

| Split | Years | Rows | Purpose |
|-------|-------|------|---------|
| Train | 2018–2022 | ~31.1M | Model fitting |
| Validation | 2023 | ~6.7M | Threshold tuning, early stopping |
| Test | 2024 | ~6.9M | Final unbiased evaluation |

---

## Repository Structure

```
AirlineArrivalDelay/
├── notebooks/
│   ├── 01_pipeline.ipynb             # BTS download, weather join (T-2h), holiday flags, Spark temporal split → parquet
│   ├── 02_eda.ipynb                  # Temporal EDA (STL, rolling trends, COVID annotation) + categorical EDA (airline, airport, route)
│   ├── 03_feature_engineering.ipynb  # Rolling delay rates (30d/90d), target encoding, temporal split export
│   ├── 04_download_data.ipynb        # One-time download of train/val/test parquet files from Zenodo to Google Drive
│   ├── 05_logistic_regression.ipynb  # Baseline logistic regression model (PySpark MLlib)
│   ├── 06_random_forest.ipynb        # Random Forest model (PySpark MLlib, 100 trees, depth 12, threshold 0.30)
│   └── 07_gradient_boosting.ipynb    # XGBoost model (SparkXGBClassifier)
├── docs/                             # EDA figures, model output plots, and writeups
├── data/                             # Gitignored — parquet files (train/val/test)
├── requirements.txt
└── README.md
```

---

## Data Pipeline

### 1. Flight Data (BTS)
- U.S. BTS On-Time Performance database — monthly CSV files, 2018–2024
- Fields retained: scheduled times, carrier, origin/destination, distance, `ArrDel15` target

### 2. Weather Data (Iowa Mesonet ASOS)
- Hourly observations at ~350 U.S. airports
- Features: temperature, dewpoint, humidity, wind speed, visibility, precipitation, weather codes
- Binary flags: `is_rain`, `is_snow`, `is_fog`, `low_visibility`, `high_wind`, `severe_weather`

### 3. Leak-Safe Weather Join
- Weather attached at **scheduled departure − 2 hours** for both origin and destination
- Ensures only information available at prediction time is used

### 4. Feature Engineering
- **Temporal:** `dep_hour`, `arr_hour`, `Month`, `DayOfWeek`, `Quarter`
- **Calendar:** `is_weekend`, `is_holiday` (±1 day window)
- **Rolling delay rates (train data only):** `carrier_delay_rate_30d/90d`, `origin_delay_rate_30d/90d`, `dest_delay_rate_30d/90d`
- **Target encoding:** origin & destination airports (384 unique values each)
- **Congestion:** `origin_departures_3h`

### 5. Leakage Controls
- 10 identified leakage risks audited and mitigated
- Strict year-based temporal splits — no shuffling
- Encoders fit on training data only, applied to val/test

---

## Models

| Notebook | Model | Framework | Key Config |
|----------|-------|-----------|-----------|
| `05_logistic_regression.ipynb` | Logistic Regression | PySpark MLlib | Baseline, class-balanced |
| `06_random_forest.ipynb` | Random Forest | PySpark MLlib | 100 trees, depth 12, threshold 0.30 |
| `07_gradient_boosting.ipynb` | Gradient Boosting | SparkXGBClassifier | 3000 rounds, depth 10, lr 0.03, early stopping |

---

## Results

| Model | Val ROC-AUC | Test PR-AUC | Test Recall | Test F1 |
|-------|------------|------------|------------|--------|
| Logistic Regression | 0.6766 | 0.3558 | 0.5469 | 0.4071 |
| Random Forest | 0.6910 | 0.3813 | 0.9278 | 0.3764 |
| XGBoost | 0.7133 | 0.4108 | 0.5739 | 0.4393 |

---

## How to Run

All notebooks were developed and run on **Google Colab Pro** with a high-RAM runtime (35GB+). Local execution is not recommended due to dataset size.

### Runtime Requirements

| Notebook | Runtime Type |
|----------|-------------|
| `05_logistic_regression.ipynb` | High-RAM CPU |
| `06_random_forest.ipynb` | High-RAM CPU |
| `07_gradient_boosting.ipynb` | High-RAM CPU + GPU |

### Steps

1. **Download the data (one time only)**
   - Open `04_download_data.ipynb` in Google Colab
   - Select a standard CPU runtime
   - Run all cells — this downloads `train.parquet`, `val.parquet`, `test.parquet` from Zenodo into `/content/drive/MyDrive/flight_data/` on your Google Drive

2. **Run the model notebooks**
   - Open `05_logistic_regression.ipynb`, `06_random_forest.ipynb`, or `07_gradient_boosting.ipynb` in Google Colab
   - Select the appropriate runtime (see table above)
   - Run all cells top to bottom — each notebook mounts Drive and reads from `flight_data/` automatically

---

## Team

- **Angela Watson** — Data pipeline, weather join, XGBoost modeling
- **Cameron Hensley** — Temporal EDA, feature engineering, logistic regression
- **Sripriya Panchapakesan** — Categorical EDA, target encoding, Random Forest modeling
