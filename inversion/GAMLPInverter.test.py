import unittest

from inversion.GAMLPInverter import GAMLPInverter
from sklearn.neural_network import MLPRegressor
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_something_else(self):
        self.assertEqual(True, True)

    def test_init_ga_popullation(self):
        X = np.floor(np.random.random((3, 3)) * 10)

        self.assertEqual(True, True)
        y = [ [np.sum(x), np.mean(x)] for x in X]
        regressor = MLPRegressor()
        regressor.fit(X,y)
        inverter = GAMLPInverter(regressor)
        print(inverter._init_ga_population())



if __name__ == '__main__':
    unittest.main()
