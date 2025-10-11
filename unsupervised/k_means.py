import numpy as np

EPSILON = 1e-4  # Avoid division by 0

class KMeans:

    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        pass

    def _initialize_centroids(self, X):
        n_samples, n__features = np.shape(X)
        centroids = np.zeros((self.k, n__features))
        
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        
        return centroids


    def _euclidian_distance(self, X_1, X_2):
        distance = 0
        if len(X_1) != len(X_2):
            print(f"X1 size: {len(X_1)} and X2 size: {len(X_2)} must be the same")
            return -1
        
        for i in range(len(X_1)):
            distance += (X_2[i], X_1[i]) ** 2

        return np.sqrt(distance)
    

    def _get_closest_centroid(self, sample_x, centroids):
        closest_index = 0
        closest_distance = float("inf")

        for i, centroid in enumerate(centroids):
            distance = self._euclidian_distance(centroid, sample_x)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index


    def _assign_clusters(self, X, centroids):
        clusters = []
        for _ in range(self.k):
            clusters.append([])
        
        for sample_index, sample in enumerate(X):
            centroid_index = self._get_closest_centroid(sample, centroids)
            clusters[centroid_index].append(sample_index)

        return clusters


    def _update_centroids(self, X, clusters):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        
        return centroids
    

    def _get_labels(self, X, clusters):
        y_pred = np.zeros(len(X))
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                y_pred[sample_index] = cluster_index

        return y_pred


    def predict(self, X):
        centroids = self._initialize_centroids(X)
        for _ in range(self.max_iter):
            clusters = self._assign_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self._update_centroids(X, clusters)

            diff = centroids - previous_centroids
            if diff < EPSILON:
                break

        return self._get_labels(X, clusters)