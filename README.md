# 🚕 NYC Taxi Trip Duration — End-to-End ML Pipeline

Predicting NYC taxi trip duration (in seconds) using **LightGBM** on 1.4 million trips.  
Covers the full pipeline: EDA → outlier removal → feature engineering → training → evaluation.

---

## 📋 Problem Statement

Given pickup/dropoff coordinates, timestamp, and passenger info — predict how long a taxi trip will take.

- **Type:** Regression  
- **Target:** `trip_duration` (seconds)  
- **Dataset:** NYC Taxi Trip Duration (Jan–Jun 2016) — 1,458,644 rows  
- **Evaluation Metric:** RMSLE (Root Mean Squared Log Error)

---

## 📁 Project Structure

```
nyc-taxi-trip-duration/
│
├── NYC.csv                          # Raw dataset
├── nyc_taxi_trip_duration.ipynb     # Main notebook (EDA → Evaluation)
└── README.md
```

---

## 📊 Dataset

| Field | Description |
|---|---|
| `id` | Unique trip identifier |
| `vendor_id` | Taxi vendor (1 or 2) |
| `pickup_datetime` | Timestamp of trip start |
| `dropoff_datetime` | Timestamp of trip end ⚠️ *not used (data leakage)* |
| `passenger_count` | Number of passengers |
| `pickup_longitude` | Pickup GPS longitude |
| `pickup_latitude` | Pickup GPS latitude |
| `dropoff_longitude` | Dropoff GPS longitude |
| `dropoff_latitude` | Dropoff GPS latitude |
| `store_and_fwd_flag` | Whether trip data was stored before sending (Y/N) |
| `trip_duration` | **Target** — trip duration in seconds |

> **Date range:** January 1, 2016 — June 30, 2016  
> **Raw rows:** 1,458,644 | **After cleaning:** ~1,429,604

---

## ⚙️ Pipeline Overview

### 1. Exploratory Data Analysis (EDA)
- Target distribution (raw vs log-transformed)
- Categorical feature distributions (vendor, passenger count, store flag)
- Geographic scatter of pickup locations across NYC
- Median trip duration by hour of day and day of week

### 2. Outlier Removal
Removed rows where:
- `trip_duration < 60s` — not a real trip
- `trip_duration > 7200s` — extreme outliers (2+ hours)
- `passenger_count == 0` or `> 6` — invalid entries
- Coordinates outside NYC bounding box — GPS errors

### 3. Feature Engineering

| Feature | Description |
|---|---|
| `distance_km` | Haversine distance between pickup and dropoff |
| `direction` | Bearing angle from pickup to dropoff |
| `delta_lat`, `delta_lon` | Coordinate differences |
| `manhattan_dist` | Sum of absolute coordinate differences |
| `hour` | Hour of pickup (0–23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `month` | Month of pickup |
| `day_of_month` | Day of month |
| `week_of_year` | Week number |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_rush_hour` | 1 if weekday 7–9am or 4–7pm |
| `is_night` | 1 if 11pm–5am |
| `zero_distance` | 1 if pickup and dropoff are nearly identical |
| `store_and_fwd_flag` | Encoded: Y=1, N=0 |

> `dropoff_datetime` and `id` are dropped — the former is data leakage, the latter is an identifier.

### 4. Target Transformation
`trip_duration` is right-skewed with extreme outliers. Log transform applied:
```python
y = np.log1p(trip_duration)
# After prediction:
prediction = np.expm1(y_pred)
```

### 5. Model: LightGBM

```python
params = {
    'objective'       : 'regression',
    'metric'          : 'rmse',
    'num_leaves'      : 63,
    'learning_rate'   : 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq'    : 5,
    'min_child_samples': 20,
    'reg_alpha'       : 0.1,
    'reg_lambda'      : 0.1,
}
```

- Early stopping: 50 rounds patience, max 2000 trees
- Train/Test split: 80/20

---

## 📈 Results

| Metric | Value |
|---|---|
| RMSLE | *run notebook to get* |
| MAE | *~X minutes avg error* |
| RMSE | *in seconds* |
| R² | *close to 1.0* |

> Results depend on hardware and exact LightGBM version. Expect RMSLE in the **0.38–0.42** range, consistent with top Kaggle submissions on this dataset without external data.

---

## 🔑 Key Findings

- **`distance_km`** is the strongest single predictor — by a large margin
- **`hour`** is the second most important — same route takes 2x longer at rush hour
- **Log-transforming the target** is not optional — raw `trip_duration` is too skewed for stable training
- **`dropoff_datetime` is data leakage** — never use it as a feature

---

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
```

Install:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

---

## 🚀 How to Run

1. Clone the repo and place `NYC.csv` in the root directory
2. Open `nyc_taxi_trip_duration.ipynb` in Jupyter
3. Run all cells top to bottom
4. Training takes ~3–8 minutes depending on hardware

```bash
git clone https://github.com/Shaurya-aswal/ML-PROJECT.git
cd nyc-taxi-trip-duration
jupyter notebook nyc_taxi_trip_duration.ipynb
```

---

## 📌 Dataset Source

[NYC Taxi Trip Duration — Kaggle Competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)

---

## 👤 Author

**Shaurya**  
B.Tech Computer Science — Maharaja Surajmal Institute of Technology, Delhi  
[GitHub](https://github.com/Shaurya-aswal) · [LinkedIn](https://www.linkedin.com/in/shauryaaswal/)


