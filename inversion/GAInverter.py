import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from inversion.WiFiRSSIPropagation import WifiRSSIPropagation
from util.util import setup_logger, calculate_spherical_coordinates

current_datetime = datetime.now()
ga_logger = setup_logger('ga_invert',
                         "log/ga_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month,
                                                         current_datetime.day,
                                                         current_datetime.hour))


class GAInverter():

    def __init__(self, index, ind_size, pop_size, elite_count, df_list_unscaled, CXPB, MUTPB, NGEN, DESIRED_OUTPUT,
                 OUTPUT_TOLERANCE, wifiRSSIPropagation_list, _DEMO_MODE):
        ga_logger.info("Instantiated GA_Inverter class")
        self.creator = creator
        self.index = index
        self.toolbox = base.Toolbox()
        self.IND_SIZE = ind_size
        self.POP_SIZE = pop_size
        self.elite_count = elite_count
        self.df_list_unscaled = df_list_unscaled
        self.wifiRSSIPropagation_list:WifiRSSIPropagation= wifiRSSIPropagation_list
        self.CXPB = CXPB;
        self.MUTPB = MUTPB;
        self.NGEN = NGEN;
        self.DESIRED_OUTPUT = DESIRED_OUTPUT;
        self.OUTPUT_TOLERANCE = OUTPUT_TOLERANCE
        self._DEMO_MODE=_DEMO_MODE
        self.__initialize_invertion_functions()

    def __creator_function(self):
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
        (r, tetha, phi) = calculate_spherical_coordinates(x, y, z)
        ga_logger.debug(
            "X: {}, Y: {}, Z: {}, X_Y: {}, X_Y_Z: {}, R: {}, Theta: {}, Phi: {}".format(x, y, z, x_y, x_y_z, r, tetha,
                                                                                        phi))
        ga_logger.info("Done generate_individual method")
        return self.wifiRSSIPropagation_list[self.index].scaler.transform([[x, y, z, x_y, x_y_z, r, tetha, phi, 0]]).tolist()[0][:-1]

    def mean_squared_error(self, individual, regressor, y_pred):
        ga_logger.info("Called evaluate method")
        #print(type(y_pred))
        if type(y_pred) is float or int:
            d = mean_squared_error((regressor.predict(np.asarray(individual).reshape(1, -1))), [y_pred])
        else:
            d = mean_squared_error((regressor.predict(np.asarray(individual).reshape(1, -1))), y_pred)
        return d

    def r2_score(self, individual, regressor, y_pred):
        ga_logger.info("Called evaluate method")
        d = r2_score((regressor.predict(np.asarray(individual).reshape(1, -1))), y_pred)
        return d

    def __initialize_invertion_functions(self):
        ga_logger.info("Started initialize_invertion_functions method")
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMin)
        # toolbox.register("attr_float", random.random())
        self.toolbox.register("population", tools.initRepeat, list, self.__creator_function, n=self.POP_SIZE)
        self.pop = self.toolbox.population()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes)
        self.toolbox.register("selectWorst", tools.selWorst)
        self.toolbox.register("selectBest", tools.selBest)
        #self.toolbox.register("score", self.mean_squared_error())
        #self.toolbox.register("score_r2", self.r2_score())
        ga_logger.info("Done initialize_invertion_functions method")

    def __initial_generation(self, y_predict, wifiRSSIPropagation):
        fitnesses = list()
        # First pass


        if self._DEMO_MODE:
            print("DEMO")
            fitnesses = self.__evaluate_individuals(wifiRSSIPropagation.model,
                                                    wifiRSSIPropagation.scaler.transform([
                                                        np.append(
                                                            np.zeros(wifiRSSIPropagation.model.coefs_[0].shape[0]),
                                                            self.DESIRED_OUTPUT)]
                                                    )[0][-1], fitnesses)
        else:
            fitnesses = self.__evaluate_individuals(wifiRSSIPropagation.model,
                                                    wifiRSSIPropagation.scaler.transform(
                                                        np.append(
                                                            np.zeros(wifiRSSIPropagation.model.coefs_[0].shape[0]),
                                                            self.DESIRED_OUTPUT)
                                                    )[0][-1], fitnesses)
        return fitnesses

    def __generate_inverted_population(self, y_predict, wifiRSSIPropagation):
        ga_logger.info("Started __generate_inverted_population method")
        fitnesses = self.__initial_generation(y_predict, wifiRSSIPropagation)
        ga_logger.debug("Initial fitness values {}".format(fitnesses))
        optimized_pop = self.__optimize_population(y_predict, wifiRSSIPropagation)
        ga_logger.info("Done __generate_inverted_population method")
        return optimized_pop

    def __create_new_generation(self, elites, offsprings):
        ga_logger.info("Started __create_new_generation method")
        for i in range(self.elite_count):
            self.pop[i] = elites[i]
        for i in range(self.POP_SIZE - self.elite_count):
            self.pop[i + self.elite_count] = offsprings[i]
        ga_logger.info("Done __create_new_generation method")

    def __generate_offspings(self, offsprings):
        ga_logger.info("Started __generate_offspings method")
        for index, offspring in enumerate(offsprings):
            #TODO BUG HERE: CHOOSES 2 ELEMENT FROM 1 ELEMENT LONG LIST IN DEMO MODE
            parents = tools.selRoulette(self.pop, 2)
            parents = list(map(self.toolbox.clone, parents))
            offsprings[index] = tools.cxTwoPoint(parents[0], parents[1])[0]
            del offspring.fitness.values
        ga_logger.info("Done __generate_offspings method")
        return parents, offsprings

    def __evaluate_invalid_individuals(self, offsprings, model, outcome, fitnesses):
        ga_logger.info("Started __evaluate_invalid_individuals method")
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        for index, individual in enumerate(invalid_ind):
            fitnesses.append(self.mean_squared_error(individual, model, outcome))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        ga_logger.info("Done __evaluate_invalid_individuals method")
        return fitnesses, invalid_ind

    def __evaluate_individuals(self, model, outcome, fitnesses):
        ga_logger.info("Started __evaluate_individuals method")
        for index, individual in enumerate(self.pop):
            temp = self.mean_squared_error(individual, model, outcome)
        fitnesses.append(temp)
        if isinstance(fitnesses[0], list):
            for ind, fit in zip(self.pop, fitnesses):
                ind.fitness.values = fit
        else:
            for ind, fit in zip(self.pop, [fitnesses]):
                ind.fitness.values = fit
        ga_logger.info("Done __evaluate_individuals method")
        return fitnesses

    def invert(self, y_pred, wifiRSSIPropagation):
        ga_logger.info("Started invert method")
        inverted_pop = self.__generate_inverted_population(y_pred, wifiRSSIPropagation)
        dataset_inverted = pd.DataFrame(data=inverted_pop,
                                        columns=["pos_x", "pos_y", "pos_z", "x_y", "x_y_z", "r", "tetha", "phi"])
        # dataset_inverted = self.df_list_unscaled.copy();
        # dataset_inverted.drop(dataset_inverted.index, inplace=True)
        # for ind, row in enumerate(inverted_pop):
        #     dataset_inverted.loc[ind] = inverted_pop[ind]
        dataset_inverted['target'] = pd.Series(y_pred, index=dataset_inverted.index)
        dataset_inverted = wifiRSSIPropagation.scaler.inverse_transform(dataset_inverted)
        ga_logger.debug("Final inverted values: ".format(dataset_inverted))
        ga_logger.info("Done invert method")
        return dataset_inverted

    def __optimize_population(self, y_predict, wifiRSSIPropagation, threshold=2):
        for g in range(self.NGEN):
            elites = self.toolbox.selectBest(self.pop, k=self.elite_count)
            elites = list(map(self.toolbox.clone, elites))
            ga_logger.debug("Chosen elites: {}".format(elites))
            offsprings = self.toolbox.selectWorst(self.pop, k=self.POP_SIZE - self.elite_count)
            offsprings = list(map(self.toolbox.clone, offsprings))
            parents, offsprings = self.__generate_offspings(offsprings)
            ga_logger.debug("Generated parents: {} and offsprings: iteration {}".format(parents, offsprings))
            fitnesses, invalid_ind = self.__evaluate_invalid_individuals(offsprings, fitnesses)
            self.__create_new_generation(elites, offsprings)
            fitnesses = self.__evaluate_individuals(wifiRSSIPropagation.model, y_predict, fitnesses)
            ga_logger.debug("Fitness values at the {}. iteration {}".format(g, fitnesses))
            ga_logger.debug("{}. generation of individuals : {}".format(g, self.pop))
        ga_logger.debug("Final generation individuals : {}".format(self.pop))
        optimized_pop = [ind for ind in self.pop if ind.fitness.values[0] < threshold]
        optimized_pop.sort(key=lambda ind: ind.fitness.values[0])
        return optimized_pop
