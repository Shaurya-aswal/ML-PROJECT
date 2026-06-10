import numpy as np
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.model_trainer import ModelTrainer
from src.ml_project.components.pipeline import remove_outliers, prepare_target

if __name__ == "__main__":
    logging.info("Starting the application...")
    try:
        data_ingestion = DataIngestion()
        df = data_ingestion.initiate_data_ingestion()
        df = remove_outliers(df)
        X_train, X_test, y_train, y_test = prepare_target(df)

        train_array = np.c_[X_train.to_numpy(), y_train.to_numpy()]
        test_array = np.c_[X_test.to_numpy(), y_test.to_numpy()]

        model_trainer = ModelTrainer()
        print(f"R2 Score: {model_trainer.initiate_model_trainer(train_array, test_array)}")
    except Exception as e:
        logging.error("An error occurred in the application.")
        raise CustomException(e, sys)
