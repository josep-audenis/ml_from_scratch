import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-4

class KMeans:

    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        pass

    def _initialize_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        
        random_indices = np.random.choice(range(n_samples), size=self.k, replace=False)
        for i, idx in enumerate(random_indices):
            centroids[i] = X[idx]
        
        return centroids


    def _euclidian_distance(self, X_1, X_2):
        return np.sqrt(np.sum((X_1 - X_2) ** 2))
    

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

            diff = np.linalg.norm(centroids - previous_centroids)
            if diff < EPSILON:
                break

        return self._get_labels(X, clusters)
    
    @staticmethod
    def calculate_wcss(X, max_k=10):
        wcss_values = []
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(k=k, max_iter=100)
            labels = kmeans.predict(X)
            wcss = 0
            
            for cluster_label in range(k):
                cluster_points = X[labels == cluster_label]
                if len(cluster_points) > 0:
                    cluster_centroid = np.mean(cluster_points, axis=0)
                    distances = np.sum((cluster_points - cluster_centroid) ** 2, axis=1)
                    wcss += np.sum(distances)
            
            wcss_values.append(wcss)
        
        return wcss_values
    

    @staticmethod
    def find_optimal_k_elbow(wcss_values):
        if len(wcss_values) < 3:
            return 2
        
        first_derivatives = []
        for i in range(1, len(wcss_values)):
            derivative = wcss_values[i-1] - wcss_values[i]
            first_derivatives.append(derivative)
        
        second_derivatives = []
        for i in range(1, len(first_derivatives)):
            second_derivative = first_derivatives[i-1] - first_derivatives[i]
            second_derivatives.append(second_derivative)
        
        optimal_k = np.argmax(second_derivatives) + 2
        return optimal_k


    @staticmethod
    def plot_elbow_method(X, max_k=10):
        wcss_values = KMeans.calculate_wcss(X, max_k)
        optimal_k = KMeans.find_optimal_k_elbow(wcss_values)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), wcss_values, 'bo-', label='WCSS')
        
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        plt.plot(optimal_k, wcss_values[optimal_k-1], 'ro', markersize=10)
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title(f'Elbow Method for Optimal k (Suggested: k={optimal_k})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return wcss_values, optimal_k


if __name__ == "__main__":
    customer_data = np.array([
        [15, 39], [15, 81], [16, 6], [16, 77], [17, 40], 
        [65, 25], [65, 80], [66, 27], [67, 85], [70, 90],
        [120, 5], [120, 80], [122, 10], [125, 75], [130, 8]
    ])

    wcss_values, optimal_k = KMeans.plot_elbow_method(X=customer_data, max_k=6)
    
    print("WCSS values for k=1 to k=6:")
    for k, wcss in enumerate(wcss_values, 1):
        print(f"k={k}: WCSS = {wcss:.2f}")

    optimal_k = optimal_k
    kmeans = KMeans(k=optimal_k, max_iter=100)
    labels = kmeans.predict(customer_data)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i in range(len(customer_data)):
        plt.scatter(customer_data[i, 0], customer_data[i, 1], 
                    color=colors[int(labels[i])], s=100, alpha=0.7)

    plt.xlabel('Annual Income ($ thousands)')
    plt.ylabel('Spending Score')
    plt.title(f'Customer Segments Discovered by K-Means (k={optimal_k})')
    plt.grid(True, alpha=0.3)
    plt.show()