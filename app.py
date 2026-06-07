from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
from src.ml_project.components.data_ingestion import DataIngestion




if __name__ == "__main__":
    logging.info("Starting the application...")


    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.error("An error occurred in the application.")
        raise CustomException(e, sys)
