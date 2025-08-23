import numpy as np
import matplotlib.pyplot as plt
import random

class KMeansClustering:
    def __init__(self,k: int,max_iter: int,datapoints: np.ndarray):
        if k > len(datapoints):
            raise ValueError("K cannot be larger than the size of the dataset")
        
        self.k = k
        self.datapoints = np.asarray(datapoints, dtype=float)
        self.labels = np.array([-1 for i in range(len(datapoints))])
        self.dim = len(datapoints[0]) # number of features
        self.max_iterations = max_iter


    def cluster(self):
        def helper(centroids: np.ndarray):
            curr_iter = 0
            while curr_iter < self.max_iterations:
                changed = False
                for i in range(len(self.datapoints)):
                    min_dist = float('inf')
                    for j in range(len(centroids)):
                        dist = np.sum((self.datapoints[i] - centroids[j])**2)
                        if dist < min_dist:
                            min_dist = dist
                            if self.labels[i] != j:
                                self.labels[i] = j
                                changed = True

                if not changed:
                    break
            
                # calculate the new centroids for each cluster
                clusters_mean = np.array([np.zeros(self.dim) for i in range(self.k)])
                clusters_size = np.array([0 for i in range(self.k)])
                for i in range(len(self.datapoints)):
                    clusters_mean[self.labels[i]] += self.datapoints[i]
                    clusters_size[self.labels[i]] += 1
                for i in range(self.k):
                    if clusters_size[i] > 0:
                        clusters_mean[i] /= clusters_size[i]
                        centroids[i] = clusters_mean[i]

                curr_iter += 1


            

        #init k random centroids
        centroids = [0] * self.k
        cnt = 0
        generated = set()
        while cnt < self.k:
            rand = random.randint(0,len(self.datapoints)-1)
            if rand not in generated:
                generated.add(rand)
                centroids[cnt] = self.datapoints[rand]
                cnt += 1
        centroids = np.array(centroids)
        helper(centroids)

        return (self.datapoints,self.labels,centroids)


# testing
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3],
])

clf = KMeansClustering(2,200,X)
y,labels,centroids = clf.cluster()

plt.scatter(y[:, 0], y[:, 1], c=labels, cmap='viridis', s=50)
plt.show()





    