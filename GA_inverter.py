import math
import random
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
import datetime

from sklearn.model_selection import train_test_split

from util import setup_logger, calculate_spherical_coordinates
current_datetime = datetime.datetime.now()
ga_logger = setup_logger('ga_invert',
                      "log/ga_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month, current_datetime.day,
                                                    current_datetime.hour))


class GA_Inverter():

    def __init__(self, index,  ind_size, pop_size, elite_count,  df_list_unscaled, scaler_list,  CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
           OUTPUT_TOLERANCE):
        ga_logger.info("Instantiated GA_Inverter class")
        self.creator = creator
        self.index = index
        self.toolbox = base.Toolbox()
        self.IND_SIZE = ind_size
        self.POP_SIZE = pop_size
        self.elite_count = elite_count
        self.df_list_unscaled = df_list_unscaled
        self.scaler_list = scaler_list
        self.CXPB=CXPB;
        self.MUTPB = MUTPB;
        self.NGEN = NGEN;
        self.DESIRED_OUTPUT = DESIRED_OUTPUT;
        self.OUTPUT_TOLERANCE= OUTPUT_TOLERANCE
        self.__initialize_invertion_functions()

    def creator_function(self):
        return self.creator.Individual(self._generate_individual())

    def _generate_individual(self):
        ga_logger.info("Started generate_individual method")
        x = random.randint(math.floor(self.df_list_unscaled[self.index].min()[0]),
                           math.floor(self.df_list_unscaled[self.index].max()[0]))
        y = random.randint(math.floor(self.df_list_unscaled[self.index].min()[1]),
                           math.floor(self.df_list_unscaled[self.index].max()[1]))
        z = random.randint(math.floor(self.df_list_unscaled[self.index].min()[2]),
                           math.floor(self.df_list_unscaled[self.index].max()[2]))

        x_y = x * y
        x_y_z = x * y * z
        (r, tetha, phi)= calculate_spherical_coordinates(x, y, z)
        ga_logger.info("Done generate_individual method")
        return self.scaler_list[self.index].transform([[x, y, z, x_y, x_y_z, r, tetha, phi, 0]]).tolist()[0][:-1]

    def evaluate(self, individual, regressor, y_pred):
        ga_logger.info("Called evaluate method")
        d = (((regressor.predict(np.asarray(individual).reshape(1, -1)) - y_pred) ** 2).sum(),)
        return d

    def __initialize_invertion_functions(self):
        ga_logger.info("Started initialize_invertion_functions method")
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMin)
        # toolbox.register("attr_float", random.random())
        self.toolbox.register("population", tools.initRepeat, list, self.creator_function, n=self.POP_SIZE)
        self.pop = self.toolbox.population()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes)
        self.toolbox.register("selectWorst", tools.selWorst)
        self.toolbox.register("selectBest", tools.selBest)
        self.toolbox.register("evaluate", self.evaluate)
        ga_logger.info("Done initialize_invertion_functions method")

    def __generate_valid_pop(self, y_predict, model, scaler):
        ga_logger.info("Started generate_valid_pop method")
        fitnesses = list()
        # evaluation
        for individual in self.pop:
            temp = self.toolbox.evaluate(individual, model,
                                         scaler.transform([[0, 0, 0, 0, 0, 0, 0, 0, self.DESIRED_OUTPUT]])[0][8])
            fitnesses.append(temp)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.NGEN):

            elits = self.toolbox.selectBest(self.pop, k=self.elite_count)
            elits = list(map(self.toolbox.clone, elits))
            offsprings = self.toolbox.selectWorst(self.pop, k=self.POP_SIZE - self.elite_count)

            offsprings = list(map(self.toolbox.clone, offsprings))

            sumFitness = 0
            for ind in self.pop:
                sumFitness = sumFitness + ind.fitness.values[0]

            for offspring in offsprings:
                parents = tools.selRoulette(self.pop, 2)
                parents = list(map(self.toolbox.clone, parents))
                offspring = tools.cxTwoPoint(parents[0], parents[1])[0]
                del offspring.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
            fitnesses = list()
            for index, individual in enumerate(invalid_ind):
                fitnesses.append(self.toolbox.evaluate(individual, model, y_predict[index]))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            for i in range(self.elite_count):
                self.pop[i] = elits[i]
            for i in range(self.POP_SIZE - self.elite_count):
                self.pop[i + self.elite_count] = offsprings[i]

            for index, individual in enumerate(self.pop):
                temp = self.toolbox.evaluate(individual, model, y_predict[index])
            fitnesses.append(temp)
            for ind, fit in zip(self.pop, fitnesses):
                ind.fitness.values = fit
        ga_logger.info("Done generate_valid_pop method")
        return [ind for ind in self.pop if ind.fitness.values[0] < 2]

    def invert(self, y_pred,  scaler, df, target, model):
        ga_logger.info("Started invert method")
        valid_pop = self.__generate_valid_pop(y_pred, model, scaler)
        dataset_inverted = df.copy();
        dataset_inverted.drop(dataset_inverted.index, inplace=True)
        for ind, row in enumerate(valid_pop):
            dataset_inverted.loc[ind] = valid_pop[ind]
        dataset_inverted['target'] = pd.Series(target)
        dataset_inverted = scaler.inverse_transform(dataset_inverted)
        ga_logger.info("Done invert method")
        return dataset_inverted

    # def invert_all(df_list, df_list_unscaled, target_list, model_list, scaler_list, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
    #                OUTPUT_TOLERANCE):
    #
    #     logger.info("Started invert_all method")
    #     inverted_list = []
    #     for index, (testDataFrame, target) in enumerate(zip(df_list, target_list)):
    #         logger.debug("Current index:{}".format(index))
    #         x_train, x_test, y_train, y_test = train_test_split(testDataFrame, target)
    #         inverter = GA_inverter.GA_Inverter(0, base.Toolbox(), x_test.iloc[0].size, len(x_test.index), 10,
    #                                            df_list_unscaled, scaler_list)
    #         inverter.initialize_invertion_functions()
    #         y_pred = model_list[index].predict(x_test)
    #         valid_pop = inverter._generate_valid_pop(index, y_pred, model_list[index], scaler_list[index], CXPB, MUTPB,
    #                                                  NGEN,
    #                                                  DESIRED_OUTPUT, OUTPUT_TOLERANCE)
    #         dataset_inverted = df_list[index].copy();
    #         # dataset_original = df_list_unscaled[index].copy().values.tolist();
    #         # dataset_original_df = df_list_unscaled[index].copy()
    #         dataset_inverted.drop(dataset_inverted.index, inplace=True)
    #         for ind, row in enumerate(valid_pop):
    #             dataset_inverted.loc[ind] = valid_pop[ind]
    #         dataset_inverted['target'] = pd.Series(target_list[index])
    #         dataset_inverted = scaler_list[index].inverse_transform(dataset_inverted)
    #         inverted_list.append(dataset_inverted)
    #     logger.info("Done invert_all method")
    #     return inverted_list
