# this file will have the code to tansformation of input data like onehot encoding,standerd scaler
import sys
import os
from src.exceptions import CustomException
from src.logger import logging
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
# this is helping the to handle the missing values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_file

@dataclass
class DataTranformationConfig:
        preprocessor_obj_file_path  = os.path.join("artifacts","preprocess.pkl")

class DataTransformerClass:
        def __init__(self):
                self.data_transfer_path = DataTranformationConfig()
        def data_transfer_func(self):
                '''
                this function help transform the data to train it or test it as we want
                '''
                logging.info("data transformation preprocessor started")
                try:
                        num_features = ["reading_score","writing_score"]
                        categorical_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
                        num_pipeleine = Pipeline(
                                steps= [
                                        ("imputer",SimpleImputer(strategy="median")),
                                        ("scaler",StandardScaler())
                                ]
                        )
                        cat_pipeline = Pipeline(
                                steps = [
                                        ("imputer",SimpleImputer(strategy="most_frequent")),
                                        ("one hot encoder",OneHotEncoder())
                                ]
                        )
                        # for these two we need to craete the compine pipeline using Column_composer
                        preprocessor = ColumnTransformer(
                                [
                                        ("numerical pipeline",num_pipeleine,num_features),
                                        ("categorical pipeline",cat_pipeline,categorical_features)
                                ]
                        )
                        return preprocessor
                except Exception as e:
                        raise CustomException(e,sys)
                
        def initiate_data_transformer(self,train_data,test_data):
                try:
                        train_df = pd.read_csv(train_data)
                        print(train_df.head())
                        test_df = pd.read_csv(test_data)
                        preprocessor_obj = self.data_transfer_func()
                        logging.info("data transformation started")
                        target_feature = "math_score"
                        train_input_df = train_df.drop(columns=[target_feature],axis=1)
                        train_target_feature = train_df[target_feature]
                        test_input_df = test_df.drop(columns=[target_feature],axis=1)
                        test_target_feature =test_df[target_feature]
                        logging.info("applying preprocessing on both training and testing dataframes")
                        input_feature_train_arr = preprocessor_obj.fit_transform(train_input_df)
                        input_feature_test_arr = preprocessor_obj.transform(test_input_df)
                        train_arr = np.c_[input_feature_train_arr,np.array(train_target_feature)]
                        test_arr = np.c_[input_feature_test_arr,np.array(test_target_feature)]
                        save_file(file_path=self.data_transfer_path.preprocessor_obj_file_path,obj = preprocessor_obj)
                        return (
                                train_arr,
                                test_arr,
                                self.data_transfer_path.preprocessor_obj_file_path
                        )
                except Exception as e:
                        raise CustomException(e,sys)