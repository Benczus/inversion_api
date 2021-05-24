import unittest

import numpy as np
from keras import Input, Model
from keras.layers import Dense

from inversion import GAKerasModelInverter
from inversion.GAKerasModelInverter import GAKerasModelInverter
from inversion.test.inverter_init_test import init_default_test_inverter


def create_testing_model(neuron_config, activation_config, input_size, output_size, losses=["mae"]):
    model_input = Input(shape=input_size)
    x = Dense(neuron_config[0], activation=activation_config[0])(model_input)
    for neurons, activations in zip(neuron_config[1:], activation_config[1:]):
        x = Dense(neurons, activation=activations)(x)
    model_output = Dense(output_size, name="output")(x)
    model = Model(model_input, outputs=model_output, name="testing_model")
    model.compile(optimizer="nadam", loss=losses)
    return model


def check_inverted_y(inverted_y_list, true_y):
    n = len(inverted_y_list)
    actual_elements = 0
    for inverted_y in inverted_y_list:
        if np.isclose(inverted_y, true_y, 0.018):
            actual_elements += 1
    return (actual_elements / n) > 0.8


class GeneticAlgorithmTests(unittest.TestCase):

    def test_init_ga_popullation(self):
        population_size = 20
        individual_size = 8
        X = np.floor(np.random.random((individual_size, individual_size)) * 10)
        y = [[np.sum(x), np.mean(x)] for x in X]
        regressor = create_testing_model([100], ["linear"], np.shape(X), np.shape(y))
        regressor.fit(X, y)
        inverter = GAKerasModelInverter(regressor, population_size=population_size)
        initial_population = inverter._init_ga_population()
        self.assertEqual(initial_population.shape, (population_size, individual_size))


class InversionTests(unittest.TestCase):

    def test_init_invert(self):
        # given, when
        inverter, _ = init_default_test_inverter()
        # then
        self.assertIsInstance(inverter, GAKerasModelInverter)

    def test_seeded_inversion_exp(self):
        # given
        inverter, regressor = init_default_test_inverter()
        true_y = np.mean([n for n in range(1, 9)]) ** 2
        inverted = inverter.invert([true_y])
        # when
        inverted_y = regressor.predict(inverted)
        # then
        self.assertEqual(check_inverted_y(inverted_y, true_y), True)


if __name__ == '__main__':
    unittest.main()
