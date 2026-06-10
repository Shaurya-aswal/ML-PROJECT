# ================================================================
#  model_trainer.py
#  NYC Taxi Trip Duration — Model Trainer
#
#  What this does:
#    1. Trains 5 models (Linear, Decision Tree, RF, XGBoost, LGBM)
#    2. Runs RandomizedSearchCV for hyperparameter tuning
#       (NOT GridSearchCV — explained below)
#    3. Evaluates all models on RMSLE, RMSE, MAE, R²
#    4. Picks best model by RMSLE (correct metric for this problem)
#    5. Logs everything to MLflow via DagsHub
#    6. Saves best model to artifacts/model.pkl
#
#  WHY the old version gave poor MAE:
#    - GridSearchCV was scoring on R² (default) not RMSLE
#    - LGBM param grid had n_estimators max=200 (too low, underfits)
#    - KNN included — unusable on 1.4M rows (memory + speed)
#    - AdaBoost included — weak on large tabular regression
#    - No early stopping on LGBM/XGBoost inside CV = underfitting
# ================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import dagshub

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------
SEED = 42
CV_FOLDS = 3  # 3-fold CV — good balance for 1.4M rows
N_ITER = 10  # RandomizedSearch iterations per model
RMSLE_THRESHOLD = 0.45  # warn if best model is above this
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "model_results.csv")


# ----------------------------------------------------------------
#  DATACLASS CONFIG
# ----------------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = MODEL_PATH
    results_file_path: str = RESULTS_PATH


# ----------------------------------------------------------------
#  MODELS + PARAM DISTRIBUTIONS
#
#  Key fixes vs old version:
#  - KNN removed    → unusable on 1.4M rows (OOM / hours)
#  - AdaBoost removed → consistently weak on large tabular regression
#  - Ridge added    → better regularised baseline than plain LinearRegression
#  - RF added       → strong ensemble, trains in reasonable time
#  - XGBoost added  → competes directly with LGBM
#  - LGBM n_estimators raised to 300-1000 (was 50-200, too low)
#  - Scoring set to neg_RMSLE everywhere (was R² default — wrong metric)
#  - RandomizedSearchCV instead of GridSearchCV:
#      GridSearch tries ALL combinations → 24+ fits per model
#      RandomizedSearch tries N_ITER random combos → faster, equally good
# ----------------------------------------------------------------
MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=SEED),
    "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "XGBoost": XGBRegressor(
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    ),
    "LGBMRegressor": LGBMRegressor(
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    ),
}

PARAM_DISTRIBUTIONS = {
    # No tunable params — just fit once
    "Linear Regression": {},
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Decision Tree": {
        "max_depth": [5, 8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.8],
    },
    # RF: trained on 200k sample inside evaluate_models() — too slow on full data
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5, 0.8],
    },
    "XGBoost": {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 5, 6, 7],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0],
    },
    # LGBM: larger n_estimators range than before (was max 200 — underfitting)
    "LGBMRegressor": {
        "n_estimators": [300, 500, 700, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 6, 8, 10],
        "min_child_samples": [10, 20, 50],
        "feature_fraction": [0.7, 0.8, 0.9],
        "bagging_fraction": [0.7, 0.8, 0.9],
        "bagging_freq": [5],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
    },
}


# ----------------------------------------------------------------
#  CUSTOM RMSLE SCORER
#  This is the correct metric for NYC Taxi Trip Duration.
#  R² was wrong — a model can have high R² but terrible RMSLE.
# ----------------------------------------------------------------
def rmsle_score(y_true, y_pred):
    """RMSLE on log-space predictions (target is already log-transformed)."""
    y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)  # replace inf/nan → 0
    y_pred_clipped = np.clip(y_pred, 0.0, 11.36)  # clip to [0, log(86400)]
    return np.sqrt(mean_squared_error(y_true, y_pred_clipped))


def neg_rmsle_scorer(estimator, X, y):
    """Scorer compatible with sklearn CV — returns negative RMSLE."""
    y_pred = estimator.predict(X)
    return -rmsle_score(y, y_pred)


# ----------------------------------------------------------------
#  EVALUATE ALL MODELS
# ----------------------------------------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
    For each model:
      - If params exist → RandomizedSearchCV with neg_rmsle scoring
      - If no params    → plain fit (e.g. Linear Regression)
      - RF              → trained on 200k sample to avoid timeout
    Returns dict: {model_name: metrics_dict}
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    report = {}
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ── Sanitize all inputs at function boundary ─────────────
    # y_test/y_train with inf/nan crash every metric call.
    # This happens when log1p is applied to bad trip_duration values
    # (zeros, negatives, un-cleaned rows) in data_transformation.
    # We fix it here defensively so training always completes.
    def _sanitize(arr, name):
        n_bad = (~np.isfinite(arr)).sum()
        if n_bad > 0:
            median = float(np.nanmedian(arr))
            arr = np.where(np.isfinite(arr), arr, median)
        return arr

    y_train = _sanitize(y_train, "y_train")
    y_test = _sanitize(y_test, "y_test")

    X_train = np.where(np.isfinite(X_train), X_train, 0.0)
    X_test = np.where(np.isfinite(X_test), X_test, 0.0)

    for name, model in models.items():
        print(f"\n  ── {name} {'─'*(40 - len(name))}")
        t0 = time.time()

        param_dist = params.get(name, {})

        # Random Forest: subsample training data to avoid hours of runtime
        if name == "Random Forest":
            sample_size = min(200_000, len(X_train))
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_fit = X_train[idx]
            y_fit = y_train[idx]
            print(f"     (training on {sample_size:,} row sample — full data too slow)")
        else:
            X_fit, y_fit = X_train, y_train

        # Tune or plain fit
        if param_dist:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=N_ITER,
                scoring=neg_rmsle_scorer,  # ← RMSLE, not R²
                cv=kf,
                n_jobs=-1,
                random_state=SEED,
                refit=True,
                verbose=0,
            )
            search.fit(X_fit, y_fit)
            best = search.best_estimator_
            print(f"     Best params : {search.best_params_}")
        else:
            model.fit(X_fit, y_fit)
            best = model

        # Update the model reference so we save the tuned version
        models[name] = best

        # Evaluate on held-out test set
        y_pred = best.predict(X_test)

        # --- Safety: handle inf/nan/overflow from weak models like Linear Regression
        # Linear Regression can predict large negatives → expm1 overflows to -inf
        # Step 1: replace nan/inf with 0 (log-space, so 0 = ~1 second prediction)
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)

        # Step 2: clip log-space predictions to a safe range
        #   log1p(1) = ~0  → minimum ~1 second trip
        #   log1p(86400) = ~11.36 → maximum 24 hour trip (extreme upper bound)
        y_pred_clip = np.clip(y_pred, 0.0, 11.36)

        rmsle = rmsle_score(y_test, y_pred_clip)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_clip))
        mae = mean_absolute_error(y_test, y_pred_clip)
        r2 = r2_score(y_test, y_pred_clip)

        report[name] = {
            "RMSLE": round(rmsle, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4),
        }

        elapsed = time.time() - t0
        print(
            f"     RMSLE : {rmsle:.4f}  |  MAE(log) : {mae:.4f} "
            f"|  R² : {r2:.4f}  |  {elapsed:.1f}s"
        )

    return report


# ----------------------------------------------------------------
#  MODEL TRAINER CLASS
# ----------------------------------------------------------------
class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()
        dagshub.init(
            repo_owner="Shaurya-aswal",
            repo_name="ML-PROJECT",
            mlflow=True,
        )

    def initiate_model_trainer(self, train_array, test_array):

        # ── Coerce inputs ────────────────────────────────────
        train_array = (
            train_array.values if isinstance(train_array, pd.DataFrame) else np.array(train_array)
        )
        test_array = (
            test_array.values if isinstance(test_array, pd.DataFrame) else np.array(test_array)
        )

        if train_array.ndim == 1:
            train_array = train_array.reshape(-1, 1)
        if test_array.ndim == 1:
            test_array = test_array.reshape(-1, 1)

        if train_array.shape[1] < 2 or test_array.shape[1] < 2:
            raise ValueError("Arrays must have at least 2 columns (features + target)")

        # Last column = target (log_duration)
        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]
        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        # ── Sanitize targets & features ─────────────────────
        # y_test contains inf/nan → means log1p was applied to
        # bad trip_duration values upstream (zero, negative, or
        # un-cleaned outliers). Fix at source, but handle here too.
        for arr_name, arr in [("y_train", y_train), ("y_test", y_test)]:
            n_bad = (~np.isfinite(arr)).sum()
            if n_bad > 0:
                print(f"\n  ⚠  WARNING: {arr_name} has {n_bad} inf/nan values.")
                print(f"     Root cause: trip_duration had zeros/negatives before")
                print(f"     log1p in data_transformation. Fix outlier removal.")
                print(f"     Replacing with median to continue.\n")

        median_train = float(np.nanmedian(y_train))
        median_test = float(np.nanmedian(y_test))
        y_train = np.where(np.isfinite(y_train), y_train, median_train)
        y_test = np.where(np.isfinite(y_test), y_test, median_test)

        # Sanitize features too — inf in X silently breaks fit()
        X_train = np.where(np.isfinite(X_train), X_train, 0.0)
        X_test = np.where(np.isfinite(X_test), X_test, 0.0)

        print("\n" + "█" * 55)
        print("  MODEL TRAINER")
        print("█" * 55)
        print(f"\n  Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        print(f"  Test  : {X_test.shape[0]:,}  rows × {X_test.shape[1]} features")
        print(f"  Target: log1p(trip_duration)")
        print(f"\n  Models  : {list(MODELS.keys())}")
        print(f"  CV folds: {CV_FOLDS}  |  RandomizedSearch iters: {N_ITER}")
        print(f"  Scoring : RMSLE (lower = better)\n")

        # ── Train + evaluate all models ───────────────────────
        model_report = evaluate_models(
            X_train, y_train, X_test, y_test, MODELS, PARAM_DISTRIBUTIONS
        )

        # ── Pick best model by RMSLE ─────────────────────────
        best_name = min(model_report, key=lambda k: model_report[k]["RMSLE"])
        best_model = MODELS[best_name]
        best_metrics = model_report[best_name]

        # ── Print comparison table ────────────────────────────
        print("\n" + "=" * 65)
        print(f"  {'Model':<22} {'RMSLE':>7} {'RMSE':>8} {'MAE':>8} {'R²':>7}")
        print("  " + "-" * 63)
        for name, m in sorted(model_report.items(), key=lambda x: x[1]["RMSLE"]):
            marker = "  ← BEST" if name == best_name else ""
            print(
                f"  {name:<22} {m['RMSLE']:>7.4f} {m['RMSE']:>8.1f}"
                f" {m['MAE']:>8.1f} {m['R2']:>7.4f}{marker}"
            )
        print("=" * 65)

        # ── Threshold warning ─────────────────────────────────
        if best_metrics["RMSLE"] > RMSLE_THRESHOLD:
            print(
                f"\n  ⚠  Warning: Best RMSLE {best_metrics['RMSLE']:.4f} "
                f"is above threshold {RMSLE_THRESHOLD}."
            )
            print(f"     Consider more feature engineering or larger n_estimators.\n")

        # ── Save results CSV ──────────────────────────────────
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        results_df = pd.DataFrame(model_report).T.reset_index()
        results_df.columns = ["Model", "RMSLE", "RMSE", "MAE", "R2"]
        results_df = results_df.sort_values("RMSLE").reset_index(drop=True)
        results_df.to_csv(self.config.results_file_path, index=False)
        print(f"\n  Results saved → {self.config.results_file_path}")

        # ── Log to MLflow ─────────────────────────────────────
        with mlflow.start_run():
            mlflow.log_param("best_model_name", best_name)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("n_iter_search", N_ITER)
            mlflow.log_metric("best_rmsle", best_metrics["RMSLE"])
            mlflow.log_metric("best_rmse", best_metrics["RMSE"])
            mlflow.log_metric("best_mae", best_metrics["MAE"])
            mlflow.log_metric("best_r2", best_metrics["R2"])

            # Log all model scores
            for name, m in model_report.items():
                safe = name.lower().replace(" ", "_")
                mlflow.log_metric(f"{safe}_rmsle", m["RMSLE"])
                mlflow.log_metric(f"{safe}_r2", m["R2"])

            mlflow.log_artifact(self.config.results_file_path)

        # ── Save best model ───────────────────────────────────
        joblib.dump(best_model, self.config.trained_model_file_path)
        print(f"  Best model saved → {self.config.trained_model_file_path}")

        print(f"\n  Best model : {best_name}")
        print(f"  RMSLE      : {best_metrics['RMSLE']:.4f}")
        print(
            f"  MAE        : {best_metrics['MAE']:.0f}s  "
            f"(~{best_metrics['MAE']/60:.1f} min avg error)"
        )
        print(f"  R²         : {best_metrics['R2']:.4f}")
        print("\n" + "█" * 55 + "\n")

        return best_metrics["RMSLE"]
