import math
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools


class GA_Inverter():

    def __init__(self, index, toolbox, ind_size, pop_size, elite_count, df_list_unscaled, scaler_list):
        self.creator = creator
        self.index = index
        self.toolbox = toolbox
        self.IND_SIZE = ind_size
        self.POP_SIZE = pop_size
        self.elite_count = elite_count
        self.df_list_unscaled=df_list_unscaled
        self.scaler_list=scaler_list

    def creator_function(self):
        return self.creator.Individual(self.generate_individual())

    def generate_individual(self):
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
        return self.scaler_list[self.index].transform([[x, y, z, x_y, x_y_z, r, tetha, phi, 0]]).tolist()[0][:-1]

    def evaluate(self, individual, regressor, y_pred):
        d = (((regressor.predict(np.asarray(individual).reshape(1, -1)) - y_pred) ** 2).sum(),)
        return d

    def initialize_invertion_functions(self):
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

    def generate_valid_pop(self, index, y_predict, model, scaler, CXPB, MUTPB, NGEN, DESIRED_OUTPUT, OUTPUT_TOLERANCE,
                           ELIT_CNT=10):

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

        return [ind for ind in self.pop if ind.fitness.values[0] < 2]


