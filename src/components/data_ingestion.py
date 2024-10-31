import sys 
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sqlalchemy import create_engine
import pandas as pd

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig():
    dataset_path = os.path.join( "artifacts", "dataset.csv")
        ##--------------------Database credentials
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"


class DataIngestion():
    def __init__(self):
        self.dataconfig = DataIngestionConfig()
        

    def initiate_data_ingestion( self  ):
        logging.info("enter to data ingestion funcion")
        try:
            engine = create_engine(self.dataconfig.connection_string)
            query = "SELECT * FROM StudentPerformance"
            
            logging.info("reading database as dataframe")

            df = pd.read_sql_query( query, engine )

            logging.info("saving top 5 records")

            df.head().to_csv(self.dataconfig.dataset_path , header= True , index= False)

            logging.info("dividing the dataset")

            train_df , test_df = train_test_split(df , test_size=0.2 ,random_state=42)

            return (train_df , test_df)

        except Exception as e:
            raise CustomException(e , sys)