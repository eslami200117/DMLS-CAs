import numpy as np 
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time

FILE_PATH = './../data/data.csv'

class KMeansClustering:
    def __init__(self, X, num_clusters, comm, rank):
        self.K = num_clusters 
        self.max_iterations = 10
        self.comm = comm
        self.rank = rank
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
    
    
    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features)) 
        for idx, cluster in enumerate(clusters):
            cluster_size = len(cluster)
            cluster_total_size = comm.reduce(cluster_size, op=MPI.SUM, root=0)
            new_centroid = np.sum(X[cluster], axis=0)
            sum_centroid = comm.reduce(new_centroid, op=MPI.SUM, root=0)
            if self.rank == 0:
                new_centroid = sum_centroid/cluster_total_size
            self.comm.Bcast(new_centroid, root=0)
            centroids[idx] = new_centroid
        return centroids
    
    
    def predict_cluster(self, clusters, X):
        y_pred = [[] for _ in range(self.num_examples)]
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    
    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.savefig(f"k_means_b.jpg")
        
    
    def fit(self, X):
        if self.rank == 0:
            centroids = self.initialize_random_centroids(X)
        else:
            centroids = np.zeros((self.K, self.num_features))
    
        self.comm.Bcast(centroids, root=0)
        
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids) 
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids
            diff_all = np.empty_like(diff)
            comm.Allreduce(diff, diff_all, op=MPI.SUM)
            if not diff_all.any():
                break
        y_pred = self.predict_cluster(clusters, X)
        all_y_pred = comm.reduce(y_pred, op=MPI.SUM, root=0)
        return all_y_pred


start_time = time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = np.genfromtxt(FILE_PATH, delimiter=',')

partition = len(data)//size

new_data = [None for _ in range(size)]

for i in range(size):
    new_data[i] = data[(i*partition): (i*partition + partition)]
    
Kmeans = KMeansClustering(new_data[rank], 5, comm, rank)

y_pred = Kmeans.fit(new_data[rank])
if rank == 0:
    Kmeans.plot_fig(data, y_pred)

end_time = time()
if rank == 0:
    print(f"--- {end_time - start_time} seconds ---")

