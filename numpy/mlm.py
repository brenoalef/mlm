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
        idx = np.random.permutation(X.shape[0])[:k]
        self.R = X[idx]
        self.T = Y[idx]

    def train(self, X, Y, k=0.5, srp='rand'):
        if Y.ndim == 1:
            Y = Y.reshape((-1, 1))
        if srp == 'kmedoids':
            self.srpKNearestCenter(X, Y, k)
        else:
            self.srpRand(X, Y, k)
        #Dx = euclidean_distances(X, self.R)
        #Dy = euclidean_distances(Y, self.T)
        #self.B_hat = np.linalg.pinv(Dx).dot(Dy)
        self.B_hat = np.linalg.pinv(euclidean_distances(X, self.R)).dot(euclidean_distances(Y, self.T))

    def predict(self, X, Y=None, method="nn"):
        #Dx = euclidean_distances(X, self.R)
        #Dy = Dx.dot(self.B_hat)
        Dy = euclidean_distances(X, self.R).dot(self.B_hat)
        if method == "nn":
            y_hat = self.T[np.argmin(Dy, axis=1)]
        elif method == "lm":
            yh0 = np.mean(self.T, axis=0)
            y_hat = np.zeros((Dy.shape[0], self.T.shape[1]), dtype=np.float64)
            for i in range(Dy.shape[0]):
                J = lambda x: np.sum(np.square(self.T - repmat(x, self.T.shape[0], 1)), 1) - np.square(Dy[i, :].T)
                pred = root(fun=J, x0=yh0, method="lm", options={"ftol":10e-6}).x
                y_hat[i] = pred
        if Y != None:
            return y_hat, (Y - y_hat)
        else:
            return y_hat       


def get_accuracy(Y, Y_hat):
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    return np.sum(np.where(Y_hat == Y, 1, 0))/Y_hat.shape[0]


def get_mse(Y, Y_hat):
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    return np.square(Y - Y_hat).mean(axis=0)


def get_amse(X, Y, model):
    Dx = euclidean_distances(X, model.R)
    Dyh = Dx.dot(model.B_hat)
    Dy = euclidean_distances(Y, model.T)
    errors = Dy - Dyh
    return np.mean(np.square(errors))
