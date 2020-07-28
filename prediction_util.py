from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import GA_inverter
from util import setup_logger

current_datetime = datetime.datetime.now()
pred_util = setup_logger('util',
                           "log/util_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                             current_datetime.day, current_datetime.hour))


def get_possible_inputs(list_of_inputs, target_list, df_list, df_list_unscaled, model_list, scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE):
    pred_util.info("Started get_output_list method")
    predicted_outputs_list = []
    for inputs_by_time in list_of_inputs:
        predicted_outputs_list.append(__predict_position(inputs_by_time, target_list, df_list, df_list_unscaled, model_list, scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE))
        pred_util.info("Done get_output_list method")
    return predicted_outputs_list


def __predict_position(inputs, target_list, df_list, df_list_unscaled, model_list, scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE):
    pred_util.info("Started predict_position method")
    output_dict = {}
    for RSSI, value in inputs.items():
        for index, target in enumerate(target_list):
            if target.name == RSSI:
                x_train, x_test, y_train, y_test = train_test_split(df_list[index], target)
                inverter = GA_inverter.GA_Inverter(index, x_test.iloc[0].size, len(x_test.index), 10,
                                                   df_list_unscaled,
                                                   scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE)
                y_pred = model_list[index].predict(x_test)
                pred_util.debug("Predicted y value:{}".format(y_pred))
                output = inverter.invert(index, y_pred,  scaler_list, df_list, target_list, model_list)
                output_dict.update({RSSI: output})
        pred_util.info("Done predict_position method")
    return output_dict


def calculate_coordinates(inverted_positions, selected_features):
    pred_util.info("Started predict_coordinates method")

    gen_x_coord = []
    gen_y_coord = []
    for values in inverted_positions.values():
        for g_val in values:
            gen_x_coord.append(g_val[0])
            gen_y_coord.append(g_val[1])
    gen_x_coord = pd.Series(gen_x_coord)
    gen_y_coord = pd.Series(gen_y_coord)
    pred_util.info("Done predict_coordinates method")
    return (pd.np.average(gen_x_coord[gen_x_coord < np.max(selected_features["pos_x"])]),
            np.average(gen_y_coord[gen_y_coord < np.max(selected_features["pos_y"])]))


def calculate_error(predicted_cooridnates, actual_coordinates):
    pred_util.info(" Called calculate_error method")
    return np.array(((predicted_cooridnates - actual_coordinates) ** 2)).mean()