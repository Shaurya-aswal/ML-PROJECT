import os
import sys
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")


def read_sql_data():
    logging.info("Reading data from MySQL database")
    try:
        mysql_connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logging.info("MySQL connection established successfully")
        df  = pd.read_sql_query("SELECT * FROM nyc", mysql_connection)
        
        return df
            
    except Exception as e:
        raise CustomException(e, sys)
    