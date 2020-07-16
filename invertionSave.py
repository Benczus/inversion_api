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

from deap import base
from deap import creator
from deap import tools  
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
dataset = pd.read_csv("dataset.csv", sep=";")


# # Imports and settings

# In[37]:


#Utility Methods

def _calculate_spherical_coordinates(dataset) : 
    r=dataset["Position X"]**2+dataset["Position Y"]**2+dataset["Position Z"]**2
    r=np.sqrt(r)
    tetha=dataset["Position Y"]/r
    tetha=np.arccos(tetha)
    phi=dataset["Position Y"]/dataset["Position X"]
    phi=np.tanh(phi)
    return (r,tetha,phi)

def create_synthetic_features(dataset):
    x_y=dataset["Position X"]*dataset["Position Y"]
    x_y_z=dataset["Position X"]*dataset["Position Y"]*dataset["Position Z"]
    (r,tetha,phi)=_calculate_spherical_coordinates(dataset)
    synthetic= pd.DataFrame()
    synthetic["x_y"]=x_y
    synthetic["x_y_z"]=x_y_z
    synthetic["r"]=r
    synthetic["tetha"]=tetha
    synthetic["phi"]=phi
    return(synthetic)

def get_AP_dataframe(selected_features, AP_name):
    AP_df=selected_features.iloc[:,0:8]
    AP_df[AP_name]=selected_features[AP_name]
    AP_df= AP_df[pd.notnull(AP_df[AP_name])]
    return AP_df
    
def get_AP_scaler(AP_df):
    scaler=preprocessing.StandardScaler()
    scaler.fit(AP_df)
    return scaler


# In[41]:


def create_ANN_list():
    ANN_List=[]
    for (testDataFrame, target) in zip(df_list,target_list):
        x_train, x_test, y_train, y_test= train_test_split(testDataFrame, target)
        model= MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(200, 300, 400, 300, 200), learning_rate='adaptive',
               learning_rate_init=0.0001, max_iter=5000, momentum=0.5,
               n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
               random_state=None, shuffle=True, solver='adam', tol=0.0001,
               validation_fraction=0.1, verbose=False, warm_start=False)
        model.fit(x_train, y_train)
        ANN_List.append(model)
        print(model.score(x_test, y_test))
    return ANN_List



# In[53]:


class GA_Inverter():
    
    def __init__(self, index, toolbox, ind_size, pop_size, elite_count):
        self.creator=creator
        self.index=index
        self.toolbox=toolbox
        self.IND_SIZE=ind_size
        self.POP_SIZE=pop_size
        self.elite_count=elite_count
        
    def creator_function(self):
        return self.creator.Individual(self.generate_individual())


    def generate_individual(self):
        x= random.randint(math.floor(df_list_unscaled[self.index].min()[0]),math.floor(df_list_unscaled[self.index].max()[0]))
        y = random.randint(math.floor(df_list_unscaled[self.index].min()[1]),math.floor(df_list_unscaled[self.index].max()[1]))
        z = random.randint(math.floor(df_list_unscaled[self.index].min()[2]),math.floor(df_list_unscaled[self.index].max()[2]))
        x_y=x*y
        x_y_z=x*y*z
        r=x**2+y**2+z**2
        r=np.sqrt(r)
        if r is not 0:
            tetha=y/r
        else:
             tetha=0
        tetha=np.arccos(tetha)
        if(x is not 0):
            phi=y/x
        else:
            phi=0
        phi=np.tanh(phi)
        return scaler_list[self.index].transform([[x, y, z, x_y, x_y_z, r, tetha, phi,0]]).tolist()[0][:-1]


    def evaluate(self,individual, regressor, y_pred):
        d=(((regressor.predict(np.asarray(individual).reshape(1, -1))-y_pred)**2).sum(),)
        return d

    def initialize_invertion_functions(self):
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMin)
        #toolbox.register("attr_float", random.random())
        self.toolbox.register("population", tools.initRepeat, list,  self.creator_function, n=self.POP_SIZE )
        self.pop = self.toolbox.population()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes)
        self.toolbox.register("selectWorst", tools.selWorst)
        self.toolbox.register("selectBest", tools.selBest)
        self.toolbox.register("evaluate", self.evaluate)


        
    def generate_valid_pop(self, index, y_predict, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE, ELIT_CNT = 10):
    
        fitnesses=list()
        # evaluation
        for individual in self.pop:
            #TODO USE SCALER LIST [INDEX], IT DOESNT WORK FOR SOME REASON
            temp=self.toolbox.evaluate(individual, model, scaler.transform([[0,0,0,0,0,0,0,0,DESIRED_OUTPUT]] )[0][8] )
            fitnesses.append(temp)
        for ind, fit in zip(self.pop, fitnesses):
              ind.fitness.values = fit

        for g in range(NGEN):

            elits = self.toolbox.selectBest(self.pop,k= ELIT_CNT)
            elits = list(map(self.toolbox.clone, elits))
            offsprings = self.toolbox.selectWorst(self.pop,k= self.POP_SIZE - ELIT_CNT)

            offsprings = list(map(self.toolbox.clone, offsprings))

            sumFitness = 0
            for ind in self.pop:
                sumFitness = sumFitness+ ind.fitness.values[0]

            for offspring in offsprings:
                    parents = tools.selRoulette(self.pop,2)
                    parents = list(map(self.toolbox.clone, parents))
                    offspring = tools.cxTwoPoint(parents[0],parents[1])[0]
                    del offspring.fitness.values

                # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
            fitnesses=list()
            for index, individual in enumerate(invalid_ind):
                fitnesses.append( self.toolbox.evaluate(individual, model, y_predict[index]))
            for ind, fit in zip(invalid_ind, fitnesses):
                  ind.fitness.values = fit

            for i in range(ELIT_CNT):
                self.pop[i] = elits[i]
            for i in range(self.POP_SIZE - ELIT_CNT):
                self.pop[i+ELIT_CNT] = offsprings[i]

            for index, individual in enumerate(self.pop):
                temp=self.toolbox.evaluate(individual, model, y_predict[index])
            fitnesses.append(temp)
            for ind, fit in zip(self.pop, fitnesses):
                  ind.fitness.values = fit

        return [ind for ind in self.pop if ind.fitness.values[0] <2]



# In[45]:


def invert_all(df_list, target_list, model_list, scaler_list,CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE):
        inverted_list=[]
        for index,(testDataFrame, target) in enumerate(zip(df_list,target_list)):
            print(index)
            x_train, x_test, y_train, y_test= train_test_split(testDataFrame, target)
            inverter=GA_Inverter(0, base.Toolbox(), x_test.iloc[0].size, len(x_test.index), 10)
            inverter.initialize_invertion_functions()
            y_pred=model_list[index].predict(x_test)
            valid_pop=inverter.generate_valid_pop(index, y_pred, model_list[index], scaler_list[index], CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE)
            dataset_inverted=df_list[index].copy();
            dataset_original=df_list_unscaled[index].copy().values.tolist();
            dataset_original_df=df_list_unscaled[index].copy()
            dataset_inverted.drop(dataset_inverted.index, inplace=True)
            for ind, row in enumerate(valid_pop):
                dataset_inverted.loc[ind] = valid_pop[ind]
            dataset_inverted['target']= pd.Series(target_list[index])
            dataset_inverted=scaler_list[index].inverse_transform(dataset_inverted)
            inverted_list.append(dataset_inverted)
        return inverted_list


# In[12]:

import matplotlib.pyplot as plt

def plot_inverted(dataset_unscaled, dataset_inverted):
    dataset_original=dataset_unscaled.copy().values.tolist();
    dataset_original_df=dataset_unscaled.copy()
    fig, (ax1, ax2, ax3) =plt.subplots(ncols=3, nrows=1, figsize= ( 20, 6))
    x_number_list_o = [values[0] for values in dataset_original ]
    # y axis value list.
    y_number_list_o = [values[1] for values in dataset_original ]
        # Draw point based on above x, y axis values.
    ax1.scatter(x_number_list_o, y_number_list_o)
    ax1.set_xlim([0-5, dataset["Position X"].max()+5])
    ax1.set_ylim([0-5, dataset["Position Y"].max()+5])
    # Set chart title.
    ax1.title.set_text("Original coordinates of the dataset") 
    # Set x, y label text.
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    x_number_list = [values[0] for values in dataset_original if values[8] > DESIRED_OUTPUT - OUTPUT_TOLERANCE and values[8] < DESIRED_OUTPUT +OUTPUT_TOLERANCE]
    # y axis value list.
    y_number_list = [values[1] for values in dataset_original if values[8] > DESIRED_OUTPUT - OUTPUT_TOLERANCE and values[8] < DESIRED_OUTPUT +OUTPUT_TOLERANCE]
        # Draw point based on above x, y axis values.
    ax2.scatter(x_number_list, y_number_list)
    ax2.set_xlim([0-5, dataset["Position X"].max()+5])
    ax2.set_ylim([0-5, dataset["Position Y"].max()+5])
    # Set chart title.
    ax2.title.set_text("Original coordinates reduced by currently detected WiFi RSSI") 
    # Set x, y label text.
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    x_number_list = [values[0] for values in dataset_inverted ]
    # y axis value list.
    y_number_list = [values[1] for values in dataset_inverted ]
    ax3.scatter(x_number_list, y_number_list, color="r" )
    ax3.set_xlim([0-5, dataset["Position X"].max()+5])
    ax3.set_ylim([0-5, dataset["Position Y"].max()+5])
    # Set chart title.
    ax3.title.set_text("Inverted coordinates by genetic algorithm")
    # Set x, y label text.
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    plt.savefig('coordinatesinverted.pdf')
    plt.show()


# In[48]:


def invert(index, df, target, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE):
            x_train, x_test, y_train, y_test= train_test_split(df, target)
            inverter=GA_Inverter(0, base.Toolbox(), x_test.iloc[0].size, len(x_test.index), 10)
            inverter.initialize_invertion_functions()
            y_pred=model_list[index].predict(x_test)
            print(y_pred)
            valid_pop=inverter.generate_valid_pop(index, y_pred, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE)
            dataset_inverted=df_list[index].copy();
            dataset_original=df_list_unscaled[index].copy().values.tolist();
            dataset_original_df=df_list_unscaled[index].copy()
            dataset_inverted.drop(dataset_inverted.index, inplace=True)
            for ind, row in enumerate(valid_pop):
                dataset_inverted.loc[ind] = valid_pop[ind]
            dataset_inverted['target']= pd.Series(target_list[index])
            dataset_inverted=scaler_list[index].inverse_transform(dataset_inverted)
            return dataset_inverted


# In[51]:


def get_output_list(list_of_inputs,CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE):
    predicted_outputs_list=[]
    for inputs_by_time in list_of_inputs:
        predicted_outputs_list.append(predict_position(inputs_by_time,CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE))
    return predicted_outputs_list


# In[49]:


def predict_position(inputs, CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE):
    output_dict={}
    for RSSI,value in inputs.items():
        for index, target in enumerate(target_list):
            if target.name == RSSI:

                output=invert(index, df_list[index], target_list[index], model_list[index], scaler_list[index], CXPB, MUTPB, NGEN, value ,OUTPUT_TOLERANCE)
                output_dict.update({RSSI:output})
    return output_dict


# In[46]:


def predict_coordinates(inverted_positions):
    gen_x_coord=[]
    gen_y_coord=[]
    for  values in  inverted_positions.values():
        for g_val in values:
            gen_x_coord.append(g_val[0])
            gen_y_coord.append(g_val[1])
    gen_x_coord=pd.Series(gen_x_coord)
    gen_y_coord=pd.Series(gen_y_coord)
    return(np.average(gen_x_coord[gen_x_coord < np.max(selected_features["pos_x"])]),np.average(gen_y_coord[gen_y_coord < np.max(selected_features["pos_y"])]))


# In[49]:


def calculate_error(predicted_cooridnates, actual_coordinates):
    return np.array(((predicted_cooridnates - actual_coordinates) ** 2)).mean()


# # Utility Methods

# In[38]:


#selected_features= dataset["Position X", "Position Y", "Position Z",]

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


# # Feature Selection

# In[39]:


df_list=list()
scaler_list=list()
target_list=list()
df_list_unscaled=list()
i=0
for index, item in enumerate(selected_features.columns):
    #Crude but works
    if index>7:
        df_list.append(get_AP_dataframe(selected_features, AP_name=item))
        df_list_unscaled.append(get_AP_dataframe(selected_features, AP_name=item))
        scaler_list.append(get_AP_scaler(df_list[i]))
        df_list[i][:]=scaler_list[i].transform(df_list[i][:])
        target_list.append(df_list[i].pop(df_list[i].columns[-1]))
        i=i+1
with open('before.txt', 'w') as f:
    for dataframe in df_list:
       print(dataframe.describe(), file=f)


# # Data Frame and scaler list creation
# 

# In[40]:


for x in range(0, 2):
    for index,dataframe in enumerate(df_list):
        if(dataframe.size < 100):
            print(index)
            del df_list[index]
            del target_list[index]
            del scaler_list[index]
            del df_list_unscaled[index]
with open('after.txt', 'w') as f:
    for dataframe in df_list:
        print(dataframe.describe(), file=f)

model_list=create_ANN_list()
with open("model_list", "wb") as fp:
     pickle.dump(model_list, fp)
        
with open("model_list", "rb") as fp:  
        model_list = pickle.load(fp)

# # In[92]:


# #TODO GET ALL inputs for all times
# #Currently only gets the inputs at time 0!

# list_of_inputs=[]

# for index in selected_features.index:
#     inputs_list_by_time={}
#     for df in df_list_unscaled:
#         df_mod=pd.DataFrame(df.iloc[:, -1])
#         for i in range(df_mod.count()[0]):
#             if df_mod.index[i] == index:
#                 inputs_list_by_time.update({df_mod.columns[0]:df_mod.iloc[0,0]})
#     list_of_inputs.append(inputs_list_by_time)
        
    


# # In[96]:


# with open("input_lists", "wb") as fp:
#     pickle.dump(list_of_inputs, fp)


# # In[47]:


with open("input_lists", "rb") as fp:  
        list_of_inputs = pickle.load(fp)


# # # Function to generate outputs for specific input RSSI values

# # In[ ]:


CXPB, MUTPB, NGEN = 0.5, 0.1, 1000
DESIRED_OUTPUT= -80
OUTPUT_TOLERANCE= 2
output_list=get_output_list(list_of_inputs,CXPB, MUTPB, NGEN, OUTPUT_TOLERANCE)


# # In[ ]:


with open("invertedpos_list", "wb") as fp:
     pickle.dump(output_list, fp)


# # In[ ]:


with open("invertedpos_list", "rb") as fp:  
        inverted_positions_list = pickle.load(fp)


# In[ ]:


error_list=[]
for invereted_positions in inverted_positions_list:
    predicted_cooridnates=np.array(predict_coordinates(inverted_positions))
    calculate_error(predicted_cooridnates, actual_coordinates)
    error_list.append(calculate_error(predicted_cooridnates, actual_coordinates))

    
with open("error_list", "wb") as fp:
      pickle.dump(output_list, fp)


with open("error_list", "rb") as fp:  
        error_list = pickle.load(fp)

# In[ ]:




