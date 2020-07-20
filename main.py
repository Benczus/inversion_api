#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import pickle
import sklearn as sk
import numpy as np
import math
import matplotlib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import random
import util
import GA_inversion
import visualization
from deap import base
from deap import creator
from deap import tools

from ANN_training import create_ANN_list

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
dataset = pd.read_csv("dataset.csv", sep=";")


def invert_all(df_list, df_list_unscaled, target_list, model_list, scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
               OUTPUT_TOLERANCE):
    inverted_list = []
    for index, (testDataFrame, target) in enumerate(zip(df_list, target_list)):
        print(index)
        x_train, x_test, y_train, y_test = train_test_split(testDataFrame, target)
        inverter = GA_inversion.GA_Inverter(0, base.Toolbox(), x_test.iloc[0].size, len(x_test.index), 10,
                                            df_list_unscaled, scaler_list)
        inverter.initialize_invertion_functions()
        y_pred = model_list[index].predict(x_test)
        valid_pop = inverter.generate_valid_pop(index, y_pred, model_list[index], scaler_list[index], CXPB, MUTPB, NGEN,
                                                DESIRED_OUTPUT, OUTPUT_TOLERANCE)
        dataset_inverted = df_list[index].copy();
        dataset_original = df_list_unscaled[index].copy().values.tolist();
        dataset_original_df = df_list_unscaled[index].copy()
        dataset_inverted.drop(dataset_inverted.index, inplace=True)
        for ind, row in enumerate(valid_pop):
            dataset_inverted.loc[ind] = valid_pop[ind]
        dataset_inverted['target'] = pd.Series(target_list[index])
        dataset_inverted = scaler_list[index].inverse_transform(dataset_inverted)
        inverted_list.append(dataset_inverted)
    return inverted_list


def invert(index, df_list_unscaled, scaler_list, df, target, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
           OUTPUT_TOLERANCE):
    x_train, x_test, y_train, y_test = train_test_split(df, target)
    inverter = GA_inversion.GA_Inverter(0, base.Toolbox(), x_test.iloc[0].size, len(x_test.index), 10, df_list_unscaled,
                                        scaler_list)
    inverter.initialize_invertion_functions()
    y_pred = model_list[index].predict(x_test)
    print(y_pred)
    valid_pop = inverter.generate_valid_pop(index, y_pred, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
                                            OUTPUT_TOLERANCE)
    dataset_inverted = df_list[index].copy();
    dataset_original = df_list_unscaled[index].copy().values.tolist();
    dataset_original_df = df_list_unscaled[index].copy()
    dataset_inverted.drop(dataset_inverted.index, inplace=True)
    for ind, row in enumerate(valid_pop):
        dataset_inverted.loc[ind] = valid_pop[ind]
    dataset_inverted['target'] = pd.Series(target_list[index])
    dataset_inverted = scaler_list[index].inverse_transform(dataset_inverted)
    return dataset_inverted


def get_output_list(list_of_inputs,  df_list_unscaled, scaler_list, CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE):
    predicted_outputs_list = []
    for inputs_by_time in list_of_inputs:
        predicted_outputs_list.append(predict_position(inputs_by_time,  df_list_unscaled, scaler_list, CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE))
    return predicted_outputs_list


def predict_position(inputs,  df_list_unscaled, scaler_list, CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE):
    output_dict = {}
    for RSSI, value in inputs.items():
        for index, target in enumerate(target_list):
            if target.name == RSSI:
                output = invert(index, df_list[index],  df_list_unscaled, scaler_list, target_list[index], model_list[index], scaler_list[index], CXPB,
                                MUTPB, NGEN, value, OUTPUT_TOLERANCE)
                output_dict.update({RSSI: output})
    return output_dict


def predict_coordinates(inverted_positions):
    gen_x_coord = []
    gen_y_coord = []
    for values in inverted_positions.values():
        for g_val in values:
            gen_x_coord.append(g_val[0])
            gen_y_coord.append(g_val[1])
    gen_x_coord = pd.Series(gen_x_coord)
    gen_y_coord = pd.Series(gen_y_coord)
    return (np.average(gen_x_coord[gen_x_coord < np.max(selected_features["pos_x"])]),
            np.average(gen_y_coord[gen_y_coord < np.max(selected_features["pos_y"])]))


def calculate_error(predicted_cooridnates, actual_coordinates):
    return np.array(((predicted_cooridnates - actual_coordinates) ** 2)).mean()


def create_inputs_by_index(selected_features, df_list_unscaled):
    list_of_inputs = []

    for index in selected_features.index:
        inputs_list_by_time = {}
        for df in df_list_unscaled:
            df_mod = pd.DataFrame(df.iloc[:, -1])
            for i in range(df_mod.count()[0]):
                if df_mod.index[i] == index:
                    inputs_list_by_time.update({df_mod.columns[0]: df_mod.iloc[0, 0]})
        list_of_inputs.append(inputs_list_by_time)

    return list_of_inputs


def create_coordiantes_by_index(selected_features):
    actual_coordinates = []
    for index in selected_features.index:
        actual_coordinates.append([selected_features.iloc[index][0], selected_features.iloc[index][1]])
    return actual_coordinates


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
with open('before.txt', 'w') as f:
    for dataframe in df_list:
        print(dataframe.describe(), file=f)

for x in range(0, 2):
    for index, dataframe in enumerate(df_list):
        if (dataframe.size < 100):
            print(index)
            del df_list[index]
            del target_list[index]
            del scaler_list[index]
            del df_list_unscaled[index]
with open('after.txt', 'w') as f:
    for dataframe in df_list:
        print(dataframe.describe(), file=f)

# Trained ANN models are saved in "model_list" file.
# model_list=create_ANN_list(df_list, target_list)
# with open("model_list", "wb") as fp:
#     pickle.dump(model_list, fp)

with open("model_list", "rb") as fp:
    model_list = pickle.load(fp)

list_of_inputs = create_inputs_by_index(selected_features, df_list_unscaled)

with open("input_lists", "wb") as fp:
    pickle.dump(list_of_inputs, fp)

with open("input_lists", "rb") as fp:
    list_of_inputs = pickle.load(fp)

actual_coordinates = create_coordiantes_by_index(selected_features)

with open("actual_coords", "wb") as fp:
    pickle.dump(actual_coordinates, fp)

with open("actual_coords", "rb") as fp:
    actual_coordinates = pickle.load(fp)

CXPB, MUTPB, NGEN = 0.5, 0.1, 1000
DESIRED_OUTPUT = -80
OUTPUT_TOLERANCE = 2
output_list = get_output_list(list_of_inputs, df_list_unscaled, scaler_list, CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE)

with open("invertedpos_list", "wb") as fp:
    pickle.dump(output_list, fp)

with open("invertedpos_list", "rb") as fp:
    inverted_positions_list = pickle.load(fp)

inverted_positions_list = []
error_list = []
for inverted_positions in inverted_positions_list:
    predicted_cooridnates = np.array(predict_coordinates(inverted_positions))
    calculate_error(predicted_cooridnates, actual_coordinates)
    error_list.append(calculate_error(predicted_cooridnates, actual_coordinates))

with open("error_list", "wb") as fp:
    pickle.dump(output_list, fp)

with open("error_list", "rb") as fp:
    error_list = pickle.load(fp)
