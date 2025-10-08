import numpy as np

EPSILON = 1e-4  # Avoid division by 0

class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis = 0)
            self.var[c] = np.var(X_c, axis = 0)
            self.priors[c] = X_c.shape[0] / X.shape[0]


    def _gaussian_probability_density_function(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- ((x - mean) ** 2) / (2 * var + EPSILON))
        denominator = np.sqrt(2 * np.pi * var + EPSILON)
        return numerator / denominator
    

    def _predict_single(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.prior[c])
            class_conditional = np.sum(np.log(self._gaussian_probability_density_function(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    

    def predict(self, X): 
        return np.array([self._predict_single(x) for x in X])

