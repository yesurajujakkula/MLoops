from src.exceptions import CustomException
from src.logger import logging
import sys
import pickle
import os
import dill
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

def save_file(file_path,obj):
        try:
                # Save to pickle file
                os.makedirs(os.path.dirname(file_path),exist_ok=True)
                with open(file_path, "wb") as f:   # 'wb' = write in binary mode
                        dill.dump(obj, f)

        except Exception as e:
                raise CustomException(e,sys)
        
def eval_func(y_predict,y_true):
        mae = mean_absolute_error(y_true,y_predict)
        r2_square = r2_score(y_true,y_predict)
        return mae,r2_square

def select_best_model(report):
        best_model = None
        best_r2 = float("-inf")   # start with very small number
        best_mae = float("inf")   # start with very large number

        for model_name, metrics in report.items():
                r2 = metrics["test_r2_score"]
                mae = metrics["test_mae"]

                # Rule: higher R² is better, and if equal, lower MAE is better
                if (r2 > best_r2) or (r2 == best_r2 and mae < best_mae):
                        best_model = (model_name, metrics)
                        best_r2 = r2
                        best_mae = mae
        return best_model,best_r2


def load_object(file_path):
        try:
                with open(file_path, "rb") as f:   # 'rb' = read in binary mode
                        return dill.load(f)
        except Exception as e:
                raise CustomException(e,sys)