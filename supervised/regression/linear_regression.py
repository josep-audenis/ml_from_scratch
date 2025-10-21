import numpy as np

class LinearRegression:
    
    def __init__(self, n_iterations=1000, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.W = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.W) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.W -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        return np.dot(X, self.W) + self.bias