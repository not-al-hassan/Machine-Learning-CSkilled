
import numpy as np
from sklearn.cluster import KMeans


class K_Means:
    def __init__(self, k = 3, max_iter = 500):
        self.k = k
        self.max_iter = max_iter

        self.EPS = 1e-4

        # For each cluster I we will store the points belongs to that cluster
        self.clusters = {}

        # To store our centroids points
        self.centroids = []

    # This method to initialize the centroids and update them later

    def InitializationOfCentroids(self, data):
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[indices].copy().astype(float)


    def distanceCentroidAndPoints(self, centroid, point):
        return np.sum((point - centroid) ** 2)

    def Assign(self, data):
        self.clusters = { i : [] for i in range(self.k)}

        for point in data:
            distance = [self.distanceCentroidAndPoints(point, centroid) for centroid in self.centroids]
            index = np.argmin(distance)
            self.clusters[index].append(point)

    def update(self):

        for cluster_index, points in self.clusters.items():
            if len(points) > 0:
                self.centroids[cluster_index] = np.mean(points, axis = 0) # remember axis = 0


    def fit(self, data):
        self.InitializationOfCentroids(data = data)

        for _ in range(self.max_iter):
            self.Assign(data)
            prev = self.centroids.copy()
            self.update()

            if np.allclose(self.centroids, prev, rtol=1e-4):
                break

    def predict(self, data):
        prediction = []

        for point in data:
            distance = [
                self.distanceCentroidAndPoints(point, centroid)
                for centroid in self.centroids
            ]

            index = np.argmin(distance)
            prediction.append(index)

        ret = []

        for x in prediction:
            ret.append(int(x))

        return ret



if __name__ == '__main__':

    np.random.seed(42)

    data = np.array([
        [1, 1],
        [1, 2],
        [2, 1],

        [10, 10],
        [10, 11],
        [11, 10],

        [20, 20],
        [20, 21],
        [21, 20],
    ])

    model = K_Means(k=3)

    model.fit(data)

    model2 = KMeans(n_clusters=3)
    model2.fit(data)

    print("Centroids of our model :")
    print()
    print(model.centroids)
    print()
    print("Predictions of our model:")
    print()
    print(model.predict(data))

    print("-------------------------------------------------")

    print("Centroids of sklearn model :")
    print()
    print(model2.cluster_centers_)

    print()
    print("Predictions of sklearn model:")
    print()
    print(model2.predict(data))


