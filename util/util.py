import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing


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


current_datetime = datetime.now()
util_logger = setup_logger('util',
                           "log/util_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                             current_datetime.day, current_datetime.hour))


def __calculate_spherical_coordinates(dataset):
    util_logger.info("Started invert_all method")
    r = dataset["Position X"] ** 2 + dataset["Position Y"] ** 2 + dataset["Position Z"] ** 2
    r = np.sqrt(r)
    tetha = dataset["Position Y"] / r
    tetha = np.arccos(tetha)
    phi = dataset["Position Y"] / dataset["Position X"]
    phi = np.tanh(phi)
    util_logger.info("Done invert_all method")
    return (r, tetha, phi)


def calculate_spherical_coordinates(x, y, z):
    util_logger.info("Started invert_all method")
    r = x ** 2 + y ** 2 + z ** 2
    r = np.sqrt(r)
    if r is not 0:
        tetha = y / r
    else:
        tetha = 0
    tetha = np.arccos(tetha)
    if (x is not 0):
        phi = y / x
    else:
        phi = 0
    phi = np.tanh(phi)
    util_logger.info("Done invert_all method")
    return (r, tetha, phi)


def create_synthetic_features(dataset):
    util_logger.info("Started create_synthetic_features method")
    x_y = dataset["Position X"] * dataset["Position Y"]
    x_y_z = dataset["Position X"] * dataset["Position Y"] * dataset["Position Z"]
    (r, tetha, phi) = __calculate_spherical_coordinates(dataset)
    synthetic = pd.DataFrame()
    synthetic["x_y"] = x_y
    synthetic["x_y_z"] = x_y_z
    synthetic["r"] = r
    synthetic["tetha"] = tetha
    synthetic["phi"] = phi
    util_logger.info("Done create_synthetic_features method")
    return (synthetic)


def get_AP_dataframe(selected_features, AP_name):
    util_logger.info("Started get_AP_dataframe method")
    AP_df = selected_features.iloc[:, 0:8]
    AP_df[AP_name] = selected_features[AP_name]
    AP_df = AP_df[pd.notnull(AP_df[AP_name])]
    util_logger.info("Done get_AP_dataframe method")
    return AP_df


def get_AP_scaler(AP_df):
    util_logger.info("Started get_AP_scaler method")
    scaler = preprocessing.StandardScaler()
    scaler.fit(AP_df)
    util_logger.info("Done get_AP_scaler method")
    return scaler


def transform_data(dataset):
    util_logger.info("Started transform_data method")
    selected_features = dataset.iloc[:, 14:45]
    selected_features.insert(0, 'pos_x', dataset["Position X"])
    selected_features.insert(1, 'pos_y', dataset["Position Y"])
    selected_features.insert(2, 'pos_z', dataset["Position Z"])
    selected_features[selected_features.pos_z != 0]
    synthetic_features = create_synthetic_features(dataset)
    selected_features.insert(3, "x_y", synthetic_features["x_y"])
    selected_features.insert(4, "x_y_z", synthetic_features["x_y_z"])
    selected_features.insert(5, "r", synthetic_features["r"])
    selected_features.insert(6, "tetha", synthetic_features["tetha"])
    selected_features.insert(7, "phi", synthetic_features["phi"])
    util_logger.info("Done transform_data method")
    return selected_features


def create_inputs_by_index(selected_features, df_list_unscaled):
    util_logger.info("Started create_inputs_by_index method")
    list_of_inputs = []
    for index in selected_features.index:
        inputs_list_by_time = {}
        for df in df_list_unscaled:
            df_mod = pd.DataFrame(df.iloc[:, -1])
            for i in range(df_mod.count()[0]):
                if df_mod.index[i] == index:
                    inputs_list_by_time.update({df_mod.columns[0]: df_mod.iloc[0, 0]})
        list_of_inputs.append(inputs_list_by_time)
        util_logger.info("Done create_inputs_by_index method")
    return list_of_inputs


def create_coordiantes_by_index(selected_features):
    util_logger.info("Started create_coordiantes_by_index method")
    actual_coordinates = []
    for index in selected_features.index:
        actual_coordinates.append([selected_features.iloc[index][0], selected_features.iloc[index][1]])
        util_logger.info("Done create_coordiantes_by_index method")
    return actual_coordinates
