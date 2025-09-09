# i am importing all the models to know which is the best model for my data
import pandas as pd
import numpy as np
# evalution matrices as the output is a continous data we are importing these evalution matrices if categorical we need
#  binary,categorical entropy
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
#models starting
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exceptions import CustomException
from src.logger import logging
import src.utils as utils
import os,sys
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from dataclasses import dataclass

@dataclass
class ModelTraininigConfig:
        training_pickle_path:str = os.path.join("artifacts","model.pkl")

class ModelTraining:
        def __init__(self):
                self.pickle_path = ModelTraininigConfig()
        def evaluvate_models_func(self,x_train,y_train,x_test,y_test,models,params):
                logging.info("model eveluvation started to choose the best model for our data")
                report = {}
                try:
                        for model_name, model_func in models.items():
                                report[model_name] = {}
                                model_params = params[model_name]
                                gs = RandomizedSearchCV(model_func,model_params,cv=3)
                                gs.fit(x_train,y_train)
                                model_func.set_params(**gs.best_params_)
                                # feeding both train set and test set to the model
                                # training the model
                                model_func.fit(x_train,y_train)
                                y_train_pred = model_func.predict(x_train)
                                y_test_pred = model_func.predict(x_test)
                                train_mae,train_r2_score = utils.eval_func(y_train,y_train_pred)
                                test_mae,test_r2_score = utils.eval_func(y_test,y_test_pred)
                                report[model_name]["train_mae"]=train_mae
                                report[model_name]["train_r2_score"] = train_r2_score
                                report[model_name]["test_mae"]=test_mae
                                report[model_name]["test_r2_score"] = test_r2_score
                        return report
                except Exception as e:
                        raise CustomException(e,sys)
        def initiate_model_trainer(self,train_array,test_array):
                logging.info("model trainer initiated")
                try:
                        x_train,y_train,x_test,y_test = (
                                train_array[:,:-1],
                                train_array[:,-1],
                                test_array[:,:-1],
                                test_array[:,-1]
                        )
                        models = {
                                        "Lasso": Lasso(),
                                        "Ridge": Ridge(),
                                        "K-Neighbors Regressor": KNeighborsRegressor(),
                                        "Decision Tree": RandomForestRegressor(),
                                        "Random Forest Regressor": RandomForestRegressor(),
                                        "XGBRegressor": XGBRegressor(), 
                                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                                        "Linear Regression": LinearRegression(),
                                        }
                        params={
                                "Lasso": {
                                        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],   # Regularization strength
                                        'max_iter': [1000, 5000, 10000],          # Iterations for convergence
                                        'tol': [1e-3, 1e-4, 1e-5]                 # Tolerance
                                },

                                "Ridge": {
                                        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],   # Regularization strength
                                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga']
                                },

                                "K-Neighbors Regressor": {
                                        'n_neighbors': [3, 5, 7, 9, 11],          # Number of neighbors
                                        'weights': ['uniform', 'distance'],       # How weights are applied
                                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                        'p': [1, 2]                               # Distance metric (1=Manhattan, 2=Euclidean)
                                },
                                "Decision Tree": {
                                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                # 'splitter':['best','random'],
                                # 'max_features':['sqrt','log2'],
                                },
                                "Random Forest Regressor":{
                                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                
                                # 'max_features':['sqrt','log2',None],
                                'n_estimators': [8,16,32,64,128,256]
                                },
                                "Linear Regression":{},
                                "XGBRegressor":{
                                'learning_rate':[.1,.01,.05,.001],
                                'n_estimators': [8,16,32,64,128,256]
                                },
                                "CatBoosting Regressor":{
                                'depth': [6,8,10],
                                'learning_rate': [0.01, 0.05, 0.1],
                                'iterations': [30, 50, 100]
                                }
                                
                        }
                        model_reports = self.evaluvate_models_func(x_train,y_train,x_test,y_test,models,params=params)
                        best_model,best_r2 = utils.select_best_model(model_reports)
                        best_model_var= models[best_model[0]]
                        if best_r2 < 0.8:
                                raise CustomException("no best model found")
                        logging.info("best model found for our dataset")
                        utils.save_file(self.pickle_path.training_pickle_path,best_model_var)
                        return best_model,best_r2
                except Exception as e:
                        raise CustomException(e,sys)
