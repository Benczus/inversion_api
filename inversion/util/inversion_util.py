from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from inversion import ga_inverter
from util.util import setup_logger

current_datetime = datetime.now()
pred_util_logger = setup_logger('util',
                                "log/util_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                                  current_datetime.day, current_datetime.hour))


def get_possible_inputs(list_of_inputs, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
                        OUTPUT_TOLERANCE, target_list):
    pred_util_logger.info("Started get_output_list method")
    predicted_outputs_list = []
    for inputs_by_time in list_of_inputs:
        prediction = __predict_position(inputs_by_time, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN,
                                        DESIRED_OUTPUT, OUTPUT_TOLERANCE, target_list)
        predicted_outputs_list.append(prediction)
        pred_util_logger.info("Current prediction: {}".format(prediction))
    pred_util_logger.info("Done get_output_list method")
    return predicted_outputs_list


def __predict_position(inputs, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
                       OUTPUT_TOLERANCE, target_list):
    pred_util_logger.info("Started predict_position method")
    output_dict = {}
    for RSSI, value in inputs.items():
        for index, target in enumerate(target_list):
            if target.name == RSSI:
                x_train, x_test, y_train, y_test = train_test_split(df_list[index], target)
                inverter = ga_inverter.ga_inverter(index, x_test.iloc[0].size, len(x_test.index), 10, df_list_unscaled,
                                                   CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE, ann_comp_list)
                y_pred = ann_comp_list[index].model.predict(x_test)
                pred_util_logger.debug("Predicted y value:{}".format(y_pred))
                output = inverter.invert(y_pred, ann_comp_list[index])
                output_dict.update({RSSI: output})
        pred_util_logger.info("Done predict_position method")
    return output_dict


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
