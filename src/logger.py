# this code is used to store the logs into txt file

import logging
import os
from datetime import datetime

# at each time runnig we are going to save the log file in the in the logs folder

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),'logs')
os.makedirs(logs_path,exist_ok=True)
logs_file_path = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
        filename=logs_file_path,
        format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level= logging.INFO,
)


if __name__ == "__main__":
        logging.info("logging has started")