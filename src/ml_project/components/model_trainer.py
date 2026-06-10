import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.utils import save_object, evaluate_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Preparing training and test input data")

            # Coerce pandas DataFrame or other array-like inputs to numpy arrays
            if isinstance(train_array, pd.DataFrame):
                train_array = train_array.values
            else:
                train_array = np.array(train_array)

            if isinstance(test_array, pd.DataFrame):
                test_array = test_array.values
            else:
                test_array = np.array(test_array)

            # Ensure 2D arrays
            if train_array.ndim == 1:
                train_array = train_array.reshape(-1, 1)
            if test_array.ndim == 1:
                test_array = test_array.reshape(-1, 1)

            # Basic validation: need at least one feature + target column
            if train_array.shape[1] < 2 or test_array.shape[1] < 2:
                raise ValueError(
                    "train_array and test_array must have at least 2 columns (features + target)"
                )

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "LGBMRegressor": LGBMRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
                "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
                "LGBMRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [31, 50, 100],
                    "max_depth": [-1, 5, 10],
                },
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                },
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            best_model_score = max(model_report.values()) if model_report else float("-inf")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(
                f"Best found model on both training and testing dataset is {best_model_name}"
            )

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            logging.info(f"R2 Score: {r2_square}")
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"Mean Absolute Error: {mae}")
            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
