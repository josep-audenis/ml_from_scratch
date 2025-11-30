import numpy as np

class LinearRegression:
    
    def __init__(self, n_iterations=1000, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.W = None
        self.bias = 0
        self.X_mean = None
        self.X_std = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / self.X_std
        
        self.W = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X_normalized, self.W) + self.bias

            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.W -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_normalized = (X - self.X_mean) / self.X_std
        return np.dot(X_normalized, self.W) + self.bias

if __name__ == "__main__":
    X = np.array([[650], [785], [1200], [1500]])
    y = np.array([200000, 250000, 350000, 450000])

    model = LinearRegression(n_iterations=1000, learning_rate=0.0001)
    model.fit(X, y)

    new_house = np.array([[1000]])
    predicted_price = model.predict(new_house)
    print(f"Predicted price for a 1000 sq ft house: ${predicted_price[0]:.2f}")
    