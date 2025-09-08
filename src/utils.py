from src.exceptions import CustomException
from src.logger import logging
import sys
import pickle
import os
import dill

def save_file(file_path,obj):
        try:
                # Save to pickle file
                os.makedirs(os.path.dirname(file_path),exist_ok=True)
                with open(file_path, "wb") as f:   # 'wb' = write in binary mode
                        dill.dump(obj, f)

        except Exception as e:
                raise CustomException(e,sys)