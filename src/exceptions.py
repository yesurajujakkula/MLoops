# This code will help us to create the custom exceptions
import sys
from src.logger import logging

def error_message_details(error_code,error_details:sys):
        # this error data will contain the file name, line number, and what is error
        _,_,error_data = error_details.exc_info()
        file_name = error_data.tb_frame.f_code.co_filename
        error_message = "The error occured in this file [{0}] line number [{1}] and error messgae : [{2}]".format(file_name,error_data.tb_lineno,str(error_code))
        return error_message

class CustomException(Exception):
        def __init__(self,error_code,error_details:sys):
                # inheriting exception class constructer
                super().__init__(error_code)
                self.error_code = error_code
                self.error_message = error_message_details(self.error_code,error_details=error_details)
        def __str__(self):
                return self.error_message


if __name__ == "__main__":
        try:
                a = 1/0
        except Exception as e:
                logging.info("Divided by zero")
                raise CustomException(e,sys)