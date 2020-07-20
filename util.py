import numpy as np
import pandas as pd
from sklearn import preprocessing


def _calculate_spherical_coordinates(dataset):
    r = dataset["Position X"] ** 2 + dataset["Position Y"] ** 2 + dataset["Position Z"] ** 2
    r = np.sqrt(r)
    tetha = dataset["Position Y"] / r
    tetha = np.arccos(tetha)
    phi = dataset["Position Y"] / dataset["Position X"]
    phi = np.tanh(phi)
    return (r, tetha, phi)


def create_synthetic_features(dataset):
    x_y = dataset["Position X"] * dataset["Position Y"]
    x_y_z = dataset["Position X"] * dataset["Position Y"] * dataset["Position Z"]
    (r, tetha, phi) = _calculate_spherical_coordinates(dataset)
    synthetic = pd.DataFrame()
    synthetic["x_y"] = x_y
    synthetic["x_y_z"] = x_y_z
    synthetic["r"] = r
    synthetic["tetha"] = tetha
    synthetic["phi"] = phi
    return (synthetic)


def get_AP_dataframe(selected_features, AP_name):
    AP_df = selected_features.iloc[:, 0:8]
    AP_df[AP_name] = selected_features[AP_name]
    AP_df = AP_df[pd.notnull(AP_df[AP_name])]
    return AP_df


def get_AP_scaler(AP_df):
    scaler = preprocessing.StandardScaler()
    scaler.fit(AP_df)
    return scaler


def transform_data(dataset):
    selected_features= dataset.iloc[:,14:45]
    selected_features.insert(0,'pos_x', dataset["Position X"])
    selected_features.insert(1,'pos_y', dataset["Position Y"])
    selected_features.insert(2,'pos_z', dataset["Position Z"])
    selected_features[selected_features.pos_z != 0]
    synthetic_features=create_synthetic_features(dataset)
    selected_features.insert(3, "x_y", synthetic_features["x_y"])
    selected_features.insert(4, "x_y_z", synthetic_features["x_y_z"])
    selected_features.insert(5, "r", synthetic_features["r"])
    selected_features.insert(6, "tetha", synthetic_features["tetha"])
    selected_features.insert(7, "phi", synthetic_features["phi"])
    return selected_features