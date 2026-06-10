import os
import sys
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pandas as pd
from src.ml_project.utils import read_sql_data
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            ## reading data from MySQL database
            df = read_sql_data()

            # Drop non-numeric columns (e.g., string IDs) so models receive numeric features
            numeric_df = df.select_dtypes(include=["number"]).copy()
            dropped = set(df.columns) - set(numeric_df.columns)
            if dropped:
                logging.info(f"Dropping non-numeric columns: {sorted(list(dropped))}")

            df = numeric_df

            logging.info("Dataset read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully")

            # return numpy arrays for downstream processing (features + target)
            return train_set.to_numpy(), test_set.to_numpy()
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e, sys)
