from typing import Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import (DERIVATIVES,
                                          inplace_identity_derivative,
                                          inplace_logistic_derivative,
                                          inplace_relu_derivative,
                                          inplace_tanh_derivative)
from sklearn.utils.extmath import safe_sparse_dot

from inversion import MLPInverter


class WLKMLPInverter(MLPInverter):
    def __init__(
        self,
        input_size,
        step_size,
        regressor: MLPRegressor,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
        iteration_count=100,
        verbose=False,
    ):
        """

        :param regressor:
        :param bounds:
        :param iteration_count:
        :param verbose:
        """
        super().__init__(regressor, bounds)
        self.iteration_count = iteration_count
        self.verbose = verbose
        self.input_size = input_size
        self.step_size = step_size

    def invert(
        self,
        desired_output,
    ):
        guessedInput = np.random.rand(self.input_size)
        layer_units = [
            [self.input_size]
            + list(self.regressor.hidden_layer_sizes)
            + [self.regressor.n_outputs_]
        ]

        for j in range(0, self.iteration_count):
            activations = [guessedInput]

            for i in range(self.regressor.n_layers_ - 1):
                activations.append(
                    np.empty((guessedInput.shape[0], layer_units[0][i + 1]))
                )
            try:
                self.regressor._forward_pass(activations)
                inplace_derivative = DERIVATIVES[self.regressor.activation]
                y_pred = activations[-1]
                deltas = activations.copy()
                deltas[-1] = self._activationFunctionDerivate(
                    activations[-1], self.regressor.activation
                ) * (y_pred - desired_output)

                for i in range(1, len(activations)):
                    deltas[-i - 1] = safe_sparse_dot(
                        deltas[-i], self.regressor.coefs_[-i].T
                    )
                    deltas[-i - 1] = inplace_derivative(activations[i], deltas[i - 1])
                    if self.verbose:
                        print("#", i)
                        print(self.regressor.coefs_[-i])
                        print(deltas[-i])
                        print(self.regressor.coefs_[-i] * deltas[-i].T)
                        print((self.regressor.coefs_[-i] * deltas[-i].T).sum(axis=1))
                        print(activations[-i - 1])
                        print(
                            self._activationFunctionDerivate(
                                activations[-i - 1], self.regressor.activation
                            )
                        )
                        print(deltas[-i - 1])
                        print("-------------------")
            except Exception as e:
                msg = f"{e}"
                print(msg)
            guessedInput = guessedInput - self.step_size * deltas[0]

        return guessedInput

    def _activationFunctionDerivate(self, X, activation):
        if activation == "tanh":
            return 1.0 / (np.cosh(X) ** 2)
        if activation == "logistic":
            log_sigmoid = 1.0 / (1.0 + np.exp(-1 * X))
            return log_sigmoid * (1.0 - log_sigmoid)
        if activation == "relu":
            return [x if x > 0.0 else 0.0 for x in X]
