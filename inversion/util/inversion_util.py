import logging
from datetime import datetime

import numpy as np
import pandas as pd

current_datetime = datetime.now()
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
