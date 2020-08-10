import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from inversion.WiFiRSSIPropagation import WifiRSSIPropagation
from inversion.ann_training import create_ANN_list
from inversion.util.inversion_util import get_possible_inputs, average_xy_positions
from util import util

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

current_datetime = datetime.now()
logger = util.setup_logger('main',
                           "log/main_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                             current_datetime.day, current_datetime.hour))

clean_run = True


def main():
    dataset = pd.read_csv("data/dataset.csv", sep=";")
    selected_features = util.transform_data(dataset)

    df_list = list()
    scaler_list = list()
    target_list = list()
    df_list_unscaled = list()
    i = 0
    for index, item in enumerate(selected_features.columns):
        if index > 7:
            df_list.append(util.get_AP_dataframe(selected_features, AP_name=item))
            df_list_unscaled.append(util.get_AP_dataframe(selected_features, AP_name=item))
            scaler_list.append(util.get_AP_scaler(df_list[i]))
            df_list[i][:] = scaler_list[i].transform(df_list[i][:])
            target_list.append(df_list[i].pop(df_list[i].columns[-1]))
            i = i + 1

    for dataframe in df_list:
        logger.debug("{}".format(dataframe.describe()))

    for x in range(0, 2):
        for index, dataframe in enumerate(df_list):
            if (dataframe.size < 100):
                logger.debug("Deleted index:{}".format(index))
                del df_list[index]
                del target_list[index]
                del scaler_list[index]
                del df_list_unscaled[index]

    for dataframe in df_list:
        logger.debug("{}".format(dataframe.describe()))

    if clean_run:
        create_ANN_list(df_list, target_list, grid_search=True)


    # MODEL LIST + SCALER LIST + Target names->wifi_rssi_propagation_model list
    ann_comp_list = []
    for scaler, target in zip(scaler_list, target_list):
        ann_comp_list.append(WifiRSSIPropagation(scaler, target.name))

    if clean_run:
        list_of_inputs = util.create_inputs_by_index(selected_features, df_list_unscaled)
        with open("model/input_lists", "wb") as fp:
            pickle.dump(list_of_inputs, fp)

    with open("model/input_lists", "rb") as fp:
        list_of_inputs = pickle.load(fp)

    if clean_run:
        actual_coordinates = util.create_coordiantes_by_index(selected_features)
        with open("model/actual_coords", "wb") as fp:
            pickle.dump(actual_coordinates, fp)

    with open("model/actual_coords", "rb") as fp:
        actual_coordinates = pickle.load(fp)

    if clean_run:
        CXPB, MUTPB, NGEN = 0.5, 0.1, 1000
        DESIRED_OUTPUT = -80
        OUTPUT_TOLERANCE = 2
        output_list = get_possible_inputs(list_of_inputs, ann_comp_list, df_list, df_list_unscaled, CXPB, MUTPB, NGEN,
                                          DESIRED_OUTPUT, OUTPUT_TOLERANCE, target_list)
        with open("model/invertedpos_list", "wb") as fp:
            pickle.dump(output_list, fp)

    with open("model/invertedpos_list", "rb") as fp:
        inverted_positions_list = pickle.load(fp)

    if clean_run:
        inverted_positions_list = []
        error_list = []
        for inverted_positions in inverted_positions_list:
            predicted_cooridnates = np.array(average_xy_positions(inverted_positions))
            error_list.append((mean_squared_error(predicted_cooridnates, actual_coordinates),
                               r2_score(predicted_cooridnates, actual_coordinates)))
        with open("model/error_list", "wb") as fp:
            pickle.dump(error_list, fp)

    with open("model/error_list", "rb") as fp:
        error_list = pickle.load(fp)


if __name__ == "__main__":
    main()
