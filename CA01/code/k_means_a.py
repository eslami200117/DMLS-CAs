import numpy as np 
import matplotlib.pyplot as plt
from time import time

FILE_PATH = './../data/data.csv'


class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters 
        self.max_iterations = 10
        
        self.num_examples, self.num_features = X.shape 
        self.plot_figure = True 
        
    
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) 
        for k in range(self.K): 
            centroid = X[np.random.choice(range(self.num_examples))] 
            centroids[k] = centroid
        return centroids 
    
    
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point-centroids)**2, axis=1))
            ) 
            clusters[closest_centroid].append(point_idx)
        return clusters 
    
    
    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features)) 
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) 
            centroids[idx] = new_centroid
        return centroids
    
    
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) 
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    
    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.savefig("k_means_a.jpg")
        
    
    def fit(self, X):
        centroids = self.initialize_random_centroids(X) 
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) 
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) 
            diff = centroids - previous_centroids 
            if not diff.any():
                break
        y_pred = self.predict_cluster(clusters, X) 
        if self.plot_figure: 
            self.plot_fig(X, y_pred) 
        return y_pred


start_time = time()
data = np.genfromtxt(FILE_PATH, delimiter=',')

Kmeans = KMeansClustering(data, 5)

y_pred = Kmeans.fit(data)

end_time = time()

print(f"--- {end_time - start_time} seconds ---")
