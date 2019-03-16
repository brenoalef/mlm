import numpy as np
import matplotlib.pyplot as plt

from numpy.matlib import repmat

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

from scipy.optimize import root


class MLM:
    def srpKNearestCenter(self, X, Y, k):
        if k < 1.0:
            k = int(round(len(X)*k))
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++')
        kmeans.fit(X)
        d = pairwise_distances(kmeans.cluster_centers_, X)
        idx = np.argmin(d, axis=1)
        self.R = X[idx]
        self.T = Y[idx]

    def srpRand(self, X, Y, k):
        if k < 1.0:
            k = int(round(len(X)*k))
        idx = np.random.choice(np.arange(X.shape[0]), k, replace=False)
        self.R = X[idx]
        self.T = Y[idx]

    def train(self, X, Y, k=0.5, srp='rand'):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if srp == 'kmedoids':
            self.srpKNearestCenter(X, Y, k)
        else:
            self.srpRand(X, Y, k)
        #Dx = euclidean_distances(X, self.R)
        #Dy = euclidean_distances(Y, self.T)
        #self.B_hat = np.linalg.pinv(Dx).dot(Dy)
        self.B_hat = np.linalg.pinv(euclidean_distances(X, self.R)).dot(euclidean_distances(Y, self.T))

    def predict(self, X, method="nn"):
        #Dx = euclidean_distances(X, self.R)
        #Dy = Dx.dot(self.B_hat)
        Dy = euclidean_distances(X, self.R).dot(self.B_hat)
        if method == "nn":
            return self.T[np.argmin(Dy, axis=1)]
        elif method == "lm":
            yh0 = np.mean(self.T, axis=0)
            y_hat = np.zeros((Dy.shape[0], self.T.shape[1]))
            for i in range(Dy.shape[0]):
                J = lambda x: np.sum(np.square(self.T - repmat(x, self.T.shape[0], 1)), 1) - np.square(Dy[i, :].T)
                y_hat[i] = root(fun=J, x0=yh0, method="lm").x
            return y_hat       


def main():
    N = 1000
    TEST_SIZE = 0.33

    input_dataset = np.array([[i, i+1] for i in range(1, N, 2)])
    output_dataset = np.array([[i + (i+1)**2] for i in range(1, N, 2)])

    X_train, X_test , Y_train, Y_test = train_test_split(input_dataset, output_dataset, test_size=TEST_SIZE)

    mlm = MLM()
    mlm.train(X_train, Y_train)
    Y_hat = mlm.predict(X_test, method="lm")

    plt.plot(range(len(Y_test)), Y_test, label="Y_test")
    plt.plot(range(len(Y_hat)), Y_hat, label="Y_hat")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == "__main__":
    main()
