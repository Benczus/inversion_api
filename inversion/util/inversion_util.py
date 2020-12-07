import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current_datetime = datetime.now()

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

pred_util_logger = setup_logger('util',
                                "log/util_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                                  current_datetime.day, current_datetime.hour))





def get_possible_inputs(list_of_inputs, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
                        OUTPUT_TOLERANCE, target_list, __DEMO_MODE):
    pred_util_logger.info("Started get_output_list method")
    predicted_outputs_list = []

    if __DEMO_MODE:
        pass #TODO WRITE SINGLE INVERTION
    else:

        for inputs_by_time in list_of_inputs:
            prediction = __predict_position(inputs_by_time, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN,
                                            DESIRED_OUTPUT, OUTPUT_TOLERANCE, target_list, __DEMO_MODE)
            predicted_outputs_list.append(prediction)
            pred_util_logger.info("Current prediction: {}".format(prediction))
        pred_util_logger.info("Done get_output_list method")
    return predicted_outputs_list

#TODO
# def __predict_position(inputs, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
#                        OUTPUT_TOLERANCE, target_list, __DEMO_MODE):
#     pred_util_logger.info("Started predict_position method")
#     output_dict={}
#     if __DEMO_MODE:
#         for RSSI, value in inputs.items():
#             for index, target in enumerate(target_list):
#                 if target.name == RSSI:
#
#
#
#                     x_train, x_test, y_train, y_test = train_test_split(df_list[index], target)
#                     #TODO population size and individual size is set here!
#                     print("x_test {}".format(x_test.iloc[0].size))
#                     print("x_test_index {}".format(len(x_test.index)))
#                     inverter = GAInverter.GAInverter(index, x_test.iloc[0].size, len(x_test.index), 10, df_list_unscaled,
#                                                      CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE, ann_comp_list, __DEMO_MODE)
#
#
#
#                     y_pred = ann_comp_list[index].model.predict(x_test)
#                     pred_util_logger.debug("Predicted y value:{}".format(y_pred))
#                     output = inverter.invert(y_pred, ann_comp_list[index])
#                     output_dict.update({RSSI: output})
#             pred_util_logger.info("Done predict_position method")
#     else:
#         for RSSI, value in inputs.items():
#             for index, target in enumerate(target_list):
#                 if target.name == RSSI:
#                     x_train, x_test, y_train, y_test = train_test_split(df_list[index], target)
#                     #TODO population size and individual size is set here!
#                     print("x_test {}".format(x_test.iloc[0].size))
#                     print("x_test_index {}".format(len(x_test.index)))
#                     inverter = GAInverter.GAInverter(index, x_test.iloc[0].size, len(x_test.index), 10, df_list_unscaled,
#                                                      CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE, ann_comp_list, __DEMO_MODE)
#
#
#
#                     y_pred = ann_comp_list[index].model.predict(x_test)
#                     pred_util_logger.debug("Predicted y value:{}".format(y_pred))
#                     output = inverter.invert(y_pred, ann_comp_list[index])
#                     output_dict.update({RSSI: output})
#             pred_util_logger.info("Done predict_position method")
#     return output_dict


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
