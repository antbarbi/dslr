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


    def fit(self, x: pd.DataFrame, y: pd.DataFrame, lr: float, epochs: int, gd_type: str, batch_size: int=None):
        cost = np.zeros(epochs, dtype=float)
        for epoch in tqdm(range(epochs)):
            if gd_type == 'gd':
                loss = self.gd(x, y, lr)
            elif gd_type == 'sgd':
                loss = self.sgd(x, y, lr)
            elif gd_type == 'mbgd':
                loss = self.mbgd(x, y, lr, batch_size)
            cost[epoch] = loss
        return loss, cost


    def gd(self, x: pd.DataFrame, y: pd.DataFrame, lr: float):
        m = x.shape[0]
        y_pred = self.predict(x)

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        self.weights -= lr * (1 / m) * np.dot(x.T, y_pred - y)
        self.bias -= lr * (1 / m) * np.sum(y_pred - y)

        return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def sgd(self, x: pd.DataFrame, y: pd.DataFrame, lr: float):
        m = x.shape[0]
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x.iloc[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(m):
            xi = x_shuffled.iloc[i:i+1]
            yi = y_shuffled[i:i+1]
            
            y_pred = self.predict(xi)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            self.weights -= lr * np.dot(xi.T, y_pred - yi)
            self.bias -= lr * np.sum(y_pred - yi)
            
        return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def mbgd(self, x: pd.DataFrame, y: pd.DataFrame, lr: float, batch_size: int):
        m = x.shape[0]
        x = x.reset_index(drop=True)  # Reset the index of the DataFrame
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x.loc[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0, m, batch_size):
            xi = x_shuffled.iloc[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            
            y_pred = self.predict(xi)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            self.weights -= lr * (1 / batch_size) * np.dot(xi.T, y_pred - yi)
            self.bias -= lr * (1 / batch_size) * np.sum(y_pred - yi)
            
        y_pred_full = self.predict(x)
        y_pred_full = np.clip(y_pred_full, 1e-15, 1 - 1e-15)
        return (-1/m) * np.sum(y * np.log(y_pred_full) + (1 - y) * np.log(1 - y_pred_full))
    
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