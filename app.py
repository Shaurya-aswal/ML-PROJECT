from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Starting the application...")
    try:
        data_ingestion = DataIngestion()
        train_array, test_array = data_ingestion.initiate_data_ingestion()

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))
    except Exception as e:
        logging.error("An error occurred in the application.")
        raise CustomException(e, sys)
