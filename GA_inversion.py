import math
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import datetime

from util import setup_logger

current_datetime = datetime.datetime.now()
ga_logger = setup_logger('ga_invert',
                      "log/ga_{}_{}_{}_{}.log".format(current_datetime.year, current_datetime.month, current_datetime.day,
                                                    current_datetime.hour))


class GA_Inverter():

    def __init__(self, index, toolbox, ind_size, pop_size, elite_count, df_list_unscaled, scaler_list):
        ga_logger.info("Instantiated GA_Inverter method")
        self.creator = creator
        self.index = index
        self.toolbox = toolbox
        self.IND_SIZE = ind_size
        self.POP_SIZE = pop_size
        self.elite_count = elite_count
        self.df_list_unscaled = df_list_unscaled
        self.scaler_list = scaler_list

    def creator_function(self):
        return self.creator.Individual(self.generate_individual())

    def generate_individual(self):
        ga_logger.info("Started generate_individual method")
        x = random.randint(math.floor(self.df_list_unscaled[self.index].min()[0]),
                           math.floor(self.df_list_unscaled[self.index].max()[0]))
        y = random.randint(math.floor(self.df_list_unscaled[self.index].min()[1]),
                           math.floor(self.df_list_unscaled[self.index].max()[1]))
        z = random.randint(math.floor(self.df_list_unscaled[self.index].min()[2]),
                           math.floor(self.df_list_unscaled[self.index].max()[2]))
        x_y = x * y
        x_y_z = x * y * z
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
        ga_logger.info("Done generate_individual method")
        return self.scaler_list[self.index].transform([[x, y, z, x_y, x_y_z, r, tetha, phi, 0]]).tolist()[0][:-1]

    def evaluate(self, individual, regressor, y_pred):
        ga_logger.info("Called evaluate method")
        d = (((regressor.predict(np.asarray(individual).reshape(1, -1)) - y_pred) ** 2).sum(),)
        return d

    def initialize_invertion_functions(self):
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

    def generate_valid_pop(self, index, y_predict, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE,
                           ELIT_CNT=10):
        ga_logger.info("Started generate_valid_pop method")
        fitnesses = list()
        # evaluation
        for individual in self.pop:
            temp = self.toolbox.evaluate(individual, model,
                                         scaler.transform([[0, 0, 0, 0, 0, 0, 0, 0, DESIRED_OUTPUT]])[0][8])
            fitnesses.append(temp)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):

            elits = self.toolbox.selectBest(self.pop, k=ELIT_CNT)
            elits = list(map(self.toolbox.clone, elits))
            offsprings = self.toolbox.selectWorst(self.pop, k=self.POP_SIZE - ELIT_CNT)

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

            for i in range(ELIT_CNT):
                self.pop[i] = elits[i]
            for i in range(self.POP_SIZE - ELIT_CNT):
                self.pop[i + ELIT_CNT] = offsprings[i]

            for index, individual in enumerate(self.pop):
                temp = self.toolbox.evaluate(individual, model, y_predict[index])
            fitnesses.append(temp)
            for ind, fit in zip(self.pop, fitnesses):
                ind.fitness.values = fit
        ga_logger.info("Done generate_valid_pop method")
        return [ind for ind in self.pop if ind.fitness.values[0] < 2]
