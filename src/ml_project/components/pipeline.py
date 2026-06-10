# ================================================================
#  data_transformation.py
#  NYC Taxi Trip Duration — sklearn.pipeline.Pipeline
#
#  Pattern:
#    - Each transformation is a plain Python function
#    - Functions are wrapped with FunctionTransformer
#    - All steps assembled into one sklearn Pipeline
#    - One pipeline.fit() runs everything top to bottom
# ================================================================

import os
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2
MODEL_PATH = "/Users/apple/ML PROJECT /artifacts/nyc_taxi_pipeline.pkl"

NYC_LAT = (40.63, 40.85)
NYC_LON = (-74.05, -73.75)

FINAL_FEATURES = [
    # raw numeric
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    # datetime features
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    # spatial features
    "distance_km",
    "direction",
    "delta_lat",
    "delta_lon",
    "manhattan_dist",
    "zero_distance",
    # encoded categorical
    "store_and_fwd_flag",
]


# ================================================================
#  TRANSFORMATION FUNCTIONS
#  Each function takes a DataFrame and returns a DataFrame.
#  These are plain functions — no classes needed.
# ================================================================


# ----------------------------------------------------------------
#  FUNCTION 1 — drop_useless_columns
#  Removes id (identifier) and dropoff_datetime (data leakage).
# ----------------------------------------------------------------
def drop_useless_columns(X: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["id", "dropoff_datetime"] if c in X.columns]
    return X.drop(columns=cols_to_drop)


# ----------------------------------------------------------------
#  FUNCTION 2 — extract_datetime_features
#  Parses pickup_datetime → hour, day_of_week, month,
#  day_of_month, week_of_year, is_weekend, is_rush_hour, is_night
# ----------------------------------------------------------------
def extract_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    dt = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon … 6=Sun
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0) & (df["hour"].isin([7, 8, 9, 16, 17, 18, 19]))
    ).astype(int)
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


# ----------------------------------------------------------------
#  FUNCTION 3 — compute_spatial_features
#  Haversine distance, direction (bearing), coordinate deltas,
#  Manhattan distance proxy, zero-distance flag.
# ----------------------------------------------------------------
def compute_spatial_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    # -- Haversine distance in km --------------------------------
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    df["distance_km"] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # -- Bearing (direction of travel) --------------------------
    df["direction"] = np.degrees(
        np.arctan2(
            df["dropoff_latitude"] - df["pickup_latitude"],
            df["dropoff_longitude"] - df["pickup_longitude"],
        )
    )

    # -- Raw coordinate differences -----------------------------
    df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # -- Manhattan distance (sum of absolute deltas) ------------
    df["manhattan_dist"] = np.abs(df["delta_lat"]) + np.abs(df["delta_lon"])

    # -- Flag for trips with near-zero movement -----------------
    df["zero_distance"] = (df["distance_km"] < 0.05).astype(int)

    return df


# ----------------------------------------------------------------
#  FUNCTION 4 — encode_categorical_features
#  Binary-encodes store_and_fwd_flag:  Y → 1,  N → 0
# ----------------------------------------------------------------
def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    return df


# ----------------------------------------------------------------
#  FUNCTION 5 — select_final_features
#  Keeps only the columns the model needs, in a fixed order.
#  Drops any intermediate or raw columns that remain.
# ----------------------------------------------------------------
def select_final_features(X: pd.DataFrame) -> pd.DataFrame:
    return X[FINAL_FEATURES]


def apply_feature_transformations(X: pd.DataFrame) -> pd.DataFrame:
    X = drop_useless_columns(X)
    X = extract_datetime_features(X)
    X = compute_spatial_features(X)
    X = encode_categorical_features(X)
    X = select_final_features(X)
    return X


# ================================================================
#  BUILD PIPELINE
#  Wraps each function with FunctionTransformer,
#  then chains them into a single sklearn Pipeline ending
#  with LGBMRegressor.
# ================================================================
def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            # Step 1 — drop id and dropoff_datetime
            ("drop_columns", FunctionTransformer(drop_useless_columns)),
            # Step 2 — extract hour, day, month, flags from datetime
            ("datetime_features", FunctionTransformer(extract_datetime_features)),
            # Step 3 — compute distance, bearing, delta coords
            ("spatial_features", FunctionTransformer(compute_spatial_features)),
            # Step 4 — encode store_and_fwd_flag as 0/1
            ("encode_categoricals", FunctionTransformer(encode_categorical_features)),
            # Step 5 — keep only the 21 final model features
            ("select_features", FunctionTransformer(select_final_features)),
            # Step 6 — LightGBM regressor (predicts log_duration)
            (
                "model",
                LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=63,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    return pipeline


# ================================================================
#  HELPER FUNCTIONS  (outside pipeline — row-count changing ops)
# ================================================================


def load_data(path: str, sample: int = None) -> pd.DataFrame:
    """Load raw CSV. Optionally sample N rows for fast testing."""
    print("\n" + "=" * 55)
    print("  LOAD DATA")
    print("=" * 55)
    df = pd.read_csv(path)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Cols    : {df.shape[1]}")
    print(f"  Nulls   : {df.isnull().sum().sum()}")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"  Sampled : {len(df):,} rows")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with invalid or extreme values.
    Must run BEFORE pipeline — pipeline cannot drop rows.
    """
    print("\n" + "=" * 55)
    print("  REMOVE OUTLIERS")
    print("=" * 55)
    before = len(df)
    mask = (
        df["trip_duration"].between(60, 7200)  # 1 min – 2 hrs
        & df["passenger_count"].between(1, 6)  # valid passenger range
        & df["pickup_latitude"].between(*NYC_LAT)  # inside NYC
        & df["pickup_longitude"].between(*NYC_LON)
        & df["dropoff_latitude"].between(*NYC_LAT)
        & df["dropoff_longitude"].between(*NYC_LON)
    )
    df = df[mask].reset_index(drop=True)
    print(f"  Before  : {before:,}")
    print(f"  After   : {len(df):,}  (removed {before - len(df):,})")
    return df


def prepare_target(df: pd.DataFrame):
    """
    Apply feature engineering, log-transform target, then split into train/test.
    Returns X_train, X_test, y_train, y_test.
    """
    print("\n" + "=" * 55)
    print("  PREPARE TARGET + SPLIT")
    print("=" * 55)

    y = np.log1p(df["trip_duration"])  # log-transform — fixes right skew
    X = df.drop(columns=["trip_duration"])  # keep all other cols for pipeline
    X = apply_feature_transformations(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"  Train   : {X_train.shape[0]:,} rows")
    print(f"  Test    : {X_test.shape[0]:,} rows")
    print(f"  Target  : log1p(trip_duration) — invert with expm1 after predict")
    return X_train, X_test, y_train, y_test


def evaluate(y_true_log, y_pred_log) -> dict:
    """Compute RMSLE, RMSE, MAE, R² and print results."""
    print("\n" + "=" * 55)
    print("  EVALUATION")
    print("=" * 55)

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))

    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n  {'Metric':<8}  Value")
    print(f"  {'-'*28}")
    print(f"  {'RMSLE':<8}  {rmsle:.4f}  (Kaggle metric — lower is better)")
    print(f"  {'RMSE':<8}  {rmse:.2f} sec")
    print(f"  {'MAE':<8}  {mae:.2f} sec  (~{mae/60:.1f} min avg error)")
    print(f"  {'R²':<8}  {r2:.4f}")

    return {
        "RMSLE": round(rmsle, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
    }


def save_pipeline(pipeline, metrics: dict):
    """Save fitted pipeline to disk."""
    print("\n" + "=" * 55)
    print("  SAVE")
    print("=" * 55)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Pipeline → {MODEL_PATH}")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics  → metrics.json")


def predict_new(raw_df: pd.DataFrame) -> np.ndarray:
    """
    Load saved pipeline and predict on new raw data.
    Returns predicted trip durations in seconds.
    """
    pipeline = joblib.load(MODEL_PATH)
    return np.expm1(pipeline.predict(raw_df))


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="/Users/apple/ML PROJECT /notebook/data/raw_data.csv"
    )
    parser.add_argument("--sample", type=int, default=None, help="Subset rows for fast testing")
    args = parser.parse_args()

    np.random.seed(SEED)

    print("\n" + "█" * 55)
    print("  NYC TAXI TRIP DURATION  —  sklearn Pipeline")
    print("█" * 55)

    # ── Pre-pipeline (these remove rows so must be outside) ──
    df = load_data(args.data, args.sample)
    df = remove_outliers(df)
    X_train, X_test, y_train, y_test = prepare_target(df)

    # ── Build pipeline ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PIPELINE STRUCTURE")
    print("=" * 55)
    print("""
  raw DataFrame
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Step 1 — drop_useless_columns          │  removes id, dropoff_datetime
  │  Step 2 — extract_datetime_features     │  hour, day_of_week, flags
  │  Step 3 — compute_spatial_features      │  distance_km, direction, deltas
  │  Step 4 — encode_categorical_features   │  store_and_fwd_flag → 0/1
  │  Step 5 — select_final_features         │  keeps 21 model columns
  │  Step 6 — LGBMRegressor                 │  predicts log(trip_duration)
  └─────────────────────────────────────────┘
       │
       ▼
  prediction → np.expm1() → seconds
    """)

    pipeline = build_pipeline()

    # ── Fit — one call runs all 6 steps ──────────────────────
    print("=" * 55)
    print("  FIT PIPELINE")
    print("=" * 55)

    start = time.time()
    pipeline.fit(
        X_train,
        y_train,
        model__eval_set=[(pipeline[:-1].transform(X_test), y_test)],
        model__callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=True),
            __import__("lightgbm").log_evaluation(period=100),
        ],
    )
    print(f"\n  Done in {time.time() - start:.1f}s")

    # ── Evaluate ─────────────────────────────────────────────
    y_pred_log = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred_log)

    # ── Save ─────────────────────────────────────────────────
    save_pipeline(pipeline, metrics)

    print("\n" + "█" * 55)
    print("  PIPELINE COMPLETE")
    print("█" * 55 + "\n")


if __name__ == "__main__":
    main()
