import numpy as np
import pandas as pd
import sys as sys
from linearRegression import LinearRegression
import matplotlib.pyplot as plt

def load(path: str) -> pd.DataFrame:
    """Load a dataBase from a csv file"""
    if path is None:
        raise ValueError("Path is None")
    if not isinstance(path, str):
        raise ValueError("Path is not a string")
    data = pd.read_csv(path)
    return data

def save(save_path: str, data: pd.DataFrame):
    """Save a dataBase to a csv file"""
    if save_path is None:
        raise ValueError("Path is None")
    if not isinstance(save_path, str):
        raise ValueError("Path is not a string")
    if data is None:
        raise ValueError("Data is None")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data is not a DataFrame")
    data.to_csv(save_path, index=False, header=["weights"])

def find_hyperparameters(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Find the best hyperparameters for the model"""
    if not isinstance(X, np.ndarray):
        raise ValueError("X is not a numpy array")
    if not isinstance(y, np.ndarray):
        raise ValueError("y is not a numpy array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y do not have the same number of samples")
    best_score = 0
    best_l_rate = 0
    best_n_iter = 0
    for l_rate in np.linspace(0.001, 0.1, 10):
        for n_iter in range(100, 5000, 100):
            model = LinearRegression(l_rate, n_iter)
            model.fit(X, y)
            score = model.score(X, y)
            if score > best_score:
                best_score = score
                best_l_rate = l_rate
                best_n_iter = n_iter
    print(f"Best score: {best_score} with learning rate: {best_l_rate} and number of iterations: {best_n_iter}")
    return best_l_rate, best_n_iter


def plt_linear(X: np.ndarray, y: np.ndarray, model: LinearRegression):
    plt.scatter(X, y, color='blue', alpha=0.5, label='Données réelles')
    try:
        X_sorted = np.sort(X, axis=0)
        y_pred = model.predict(X_sorted)
        plt.plot(X_sorted, y_pred, color='red', label='Régression linéaire')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Linear Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"An error occured: {e}")

def main():
    try:
        if (len(sys.argv) != 2):
            raise ValueError("Usage: python train.py path")
        data = load(sys.argv[1])
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        l_rate, n_iter = find_hyperparameters(X, y)
        model = LinearRegression(l_rate, n_iter)
        model.fit(X, y)
        if model.weights is None:
            raise ValueError("Model is not trained")
        save("weights.csv", pd.DataFrame(model.weights))
        print ("Model is trained and weights are saved, now you can use predict.py to predict values")
        plt_linear(X, y, model)

    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
