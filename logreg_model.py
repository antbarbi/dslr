import numpy as np
import pandas as pd
from tqdm import tqdm


class LogRegModel:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.bias = 0.0


    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return 1.0 / (1.0 + np.exp(-(np.dot(x, self.weights) + self.bias)))


    def fit(self, x: pd.DataFrame, y: pd.DataFrame, lr: float, epochs: int):
        for epoch in tqdm(range(epochs)):
            loss = self.update(x, y, lr)
        return loss


    def update(self, x: pd.DataFrame, y: pd.DataFrame, lr: float):
        m = x.shape[0]
        y_pred = self.predict(x)

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        self.weights -= lr * (1 / m) * np.dot(x.T, y_pred - y)
        self.bias -= lr * (1 / m) * np.sum(y_pred - y)

        return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


    def predict_proba(self, X):
        # Sigmoid function to get probabilities
        probabilities = 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias)))
        return np.column_stack((1 - probabilities, probabilities)) 

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        y_pred = self.predict(x)
        accuracy = np.mean(y_pred == y)
        # print(f"Weights: {self.weights}, Bias: {self.bias}")
        return accuracy

    def get_weights(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias
        }
    
    def set_weights(self, weights_dict):
        self.weights = np.array(weights_dict["weights"])
        self.bias = weights_dict["bias"]