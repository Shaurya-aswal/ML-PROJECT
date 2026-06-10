import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle


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
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    logging.info("Evaluating models")
    try:

        model_report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    logging.info("Saving object to file")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)