import numpy as np

class KNN:

    def __init__(self, k):
        self.k = k


    def _euclidian_distance(self, X_1, X_2):
        distance = 0
        
        if len(X_1) != len(X_2):
            print(f"X1 size: {len(X_1)} and X2 size: {len(X_2)} must be the same")
            return -1
        
        for i in range(len(X_1)):
            distance += (X_2[i], X_1[i]) ** 2

        return np.sqrt(distance)

    
    def _get_neighbors(self, X_predict):
        distances = []

        for index, X_train in enumerate(self.X):
            distance = self._euclidian_distance(X_train, X_predict)
            distances.append((distance, index))

        distances.sort(key=lambda x: x[0])    
        neighbors = []

        for i in range(self.k):
            neighbors.append(distances[i][1])

        return neighbors
        

    def fit(self, X, y):
        self.X = X
        self.y = y


    def predict(self, X_predict):
        neighbors = self._get_neighbors(X_predict)

        neighbors_labels = []

        for i in neighbors:
            neighbors_labels.append(self.y[i])

            unique, count = np.unique(neighbors_labels, return_counts=True)
            prediction = unique[np.argmax(count)]
        
        return prediction