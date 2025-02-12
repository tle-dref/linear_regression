import numpy as np
from numpy.core.multiarray import ndarray
import pandas as pd
import sys as sys

class LinearRegression():
    def __init__(self, l_rate: float = 0.01, n_iter: int = 1000):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.weights = None
        self.tol = 1e-4
        self.mean = 0
        self.std = 0
        self.y_mean = 0
        self.y_std = 0

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Cannot call gradient before fitting the model")
        # np.dot(X, self.weights) - y) is the vector of errors, check the diff with the real value
        # np.dot(X.T, np.dot(X, self.weights) - y) is the gradient of the loss function
        # X.T is the bias column
        # we divide by y.size to normalize the values of the gradient
        return np.dot(X.T, np.dot(X, self.weights) - y) / y.size

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise ValueError("X is not a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is not a numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y do not have the same number of samples")
        # Add a column of ones to X to account for the bias
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - self.mean) / self.std
        self.y_mean = y.mean()
        self.y_std = y.std()
        y = (y - self.y_mean) / self.y_std
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            grad = self.gradient(X, y)
            newWeights = self.weights - self.l_rate * grad
            if np.all(np.abs(newWeights - self.weights) < self.tol):
                break
            # Update weights with the gradient of the loss function, multiplied by the learning rate to avoid overshooting
            self.weights = newWeights

        #  denormalize the weights and the bias
        self.weights[1:] = (self.weights[1:] * self.y_std) / self.std
        self.weights[0] = self.y_mean - np.sum(self.weights[1:] * self.mean)



    def predict_for_input(self, weights: ndarray, input: int) -> int:
        if not isinstance(input, int):
            raise ValueError("input is not an integer")
        if input < 0:
            raise ValueError("mileage is negative")
        if weights is None:
            return 0
        result = weights[0] + input * weights[1]
        if result < 0:
            return 0
        return result

    def load_weights(self):
        try:
            self.weights = pd.read_csv("weights.csv").values
        except:
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("X is not a numpy array")
        if self.weights is None:
            raise ValueError("Cannot call predict before fitting the model")
        # Add a column of ones to X to account for the bias
        X = np.insert(X, 0, 1, axis=1)
        # Return the dot product of X and the self.weights (matrix multiplication)
        return np.dot(X, self.weights)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not isinstance(X, np.ndarray):
            raise ValueError("X is not a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is not a numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y do not have the same number of samples")
        if self.weights is None:
            raise ValueError("Cannot call score before fitting the model")
        y_pred = self.predict(X)
        # Return the R^2 score
        SSR = np.sum((y - y_pred) ** 2)  # Sum of squared residuals
        SST = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        if SST == 0:
            return 0
        r2 = 1 - (SSR / SST)
        return r2
