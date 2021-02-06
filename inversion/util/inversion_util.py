import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from inversion import logger

current_datetime = datetime.now()


# def setup_logger(logger_name, log_file, level=logging.INFO):
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.DEBUG)
#     # create file handler which logs even debug messages
#     if not os.path.exists('log'):
#         os.makedirs('log')
#     fh = logging.FileHandler(log_file)
#     fh.setLevel(logging.DEBUG)
#     # create console handler with a higher log level
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     # create formatter and add it to the handlers
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)
#     # add the handlers to the logger
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     return logger


pred_util_logger = logging.getLogger("logger")

def average_xy_positions(inverted_positions, selected_features):
    pred_util_logger.info("Started predict_coordinates method")
    gen_x_coord = []
    gen_y_coord = []
    for values in inverted_positions.values():
        for g_val in values:
            gen_x_coord.append(g_val[0])
            gen_y_coord.append(g_val[1])
    gen_x_coord = pd.Series(gen_x_coord)
    gen_y_coord = pd.Series(gen_y_coord)
    pred_util_logger.info("Done predict_coordinates method")
    return (pd.np.average(gen_x_coord[gen_x_coord < np.max(selected_features["pos_x"])]),
            np.average(gen_y_coord[gen_y_coord < np.max(selected_features["pos_y"])]))
