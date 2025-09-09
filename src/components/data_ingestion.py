import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformerClass
from src.components.model_trainer import ModelTraining
#import logging

# creating the data ingestion config which will create all the required paths
@dataclass
class DataIngestionClassConfig:
        data_path:str = os.path.join("artifacts","data.csv")
        train_data_path:str = os.path.join("artifacts","train.csv")
        test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestionClass:
        def __init__(self):
                self.arti_data_path = DataIngestionClassConfig()
        def ingestion_func(self):
                # this might api,data source(data bases), for now we are taking from local path
                logging.info("Data ingestion started")
                try:
                        data_path = os.path.join(r"C:\Users\Yesuraju\Desktop\MachineLearningWorkSpace\notebook\data\stud.csv")
                        df = pd.read_csv(data_path)
                        # full data artifacts storages
                        os.makedirs(os.path.dirname(self.arti_data_path.data_path),exist_ok=True)
                        df.to_csv(self.arti_data_path.data_path,index=False,header=True)
                        logging.info("Splitting the data to train test datas")
                        train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
                        os.makedirs(os.path.dirname(self.arti_data_path.train_data_path),exist_ok=True)
                        train_data.to_csv(self.arti_data_path.train_data_path,index=False,header=True)
                        os.makedirs(os.path.dirname(self.arti_data_path.test_data_path),exist_ok=True)
                        test_data.to_csv(self.arti_data_path.test_data_path,index=False,header=True)
                        return (
                                self.arti_data_path.data_path,
                                self.arti_data_path.train_data_path,
                                self.arti_data_path.test_data_path
                        )
                except Exception as e:
                        raise CustomException(e,sys)
                
if __name__ == "__main__":
        obj = DataIngestionClass()
        data_path,train_path,test_path = obj.ingestion_func()
        data_transfer_obj = DataTransformerClass()
        train_arr,test_arr, _ = data_transfer_obj.initiate_data_transformer(train_path,test_path)
        ModelTraining_obj = ModelTraining()
        best_model,best_r2 = ModelTraining_obj.initiate_model_trainer(train_arr,test_arr)

