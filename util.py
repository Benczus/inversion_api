import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime
import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
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


current_datetime = datetime.datetime.now()
util_logger = setup_logger('util',
                           "log/util_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                         current_datetime.day, current_datetime.hour))


def _calculate_spherical_coordinates(dataset):
    util_logger.info("Started invert_all method")
    r = dataset["Position X"] ** 2 + dataset["Position Y"] ** 2 + dataset["Position Z"] ** 2
    r = np.sqrt(r)
    tetha = dataset["Position Y"] / r
    tetha = np.arccos(tetha)
    phi = dataset["Position Y"] / dataset["Position X"]
    phi = np.tanh(phi)
    util_logger.info("Done invert_all method")
    return (r, tetha, phi)


def create_synthetic_features(dataset):
    util_logger.info("Started create_synthetic_features method")
    x_y = dataset["Position X"] * dataset["Position Y"]
    x_y_z = dataset["Position X"] * dataset["Position Y"] * dataset["Position Z"]
    (r, tetha, phi) = _calculate_spherical_coordinates(dataset)
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

