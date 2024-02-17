import numpy as np

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distance = []
        for j in range(K):
            norm = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm)
        idx[i] = np.argmin(distance)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        Ck = idx == i
        centroids[i] = np.mean(X[Ck], axis = 0)
    return centroids

def run_K_means(X, initial_centroids, max_iters=10):
    centroids = initial_centroids
    K = initial_centroids.shape[0]
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids