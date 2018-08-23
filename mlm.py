import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

class MLM:
    def train(self, X, Y, k=None):
        if k == None:
            k = int(round(len(X)*0.4))
        idx = np.random.choice(np.arange(X.shape[0]), k, replace=False)
        self.R = X[idx]
        self.T = Y[idx]
        Dx = euclidean_distances(X, self.R)
        Dy = euclidean_distances(Y, self.T)
        self.B_hat = np.linalg.pinv(Dx).dot(Dy)

    def predict(self, X):
        Dx = euclidean_distances(X, self.R)
        Dy = Dx.dot(self.B_hat)
        pred_hat = []
        for i in range(len(X)):
            # TODO: Use Levenberg–Marquardt
            pred_hat.append(self.T[np.argmin(Dy[i])])
        return np.array(pred_hat)


def main():
    N = 1000
    TEST_SIZE = 0.33

    input_dataset = np.array([[i, i+1] for i in range(1, N, 2)])
    output_dataset = np.array([[i + (i+1)**2] for i in range(1, N, 2)])

    X_train, X_test , Y_train, Y_test = train_test_split(input_dataset, output_dataset, test_size=TEST_SIZE)

    mlm = MLM()
    mlm.train(X_train, Y_train)
    Y_hat = mlm.predict(X_test)

    plt.plot(range(len(Y_test)), Y_test, label="Y_test")
    plt.plot(range(len(Y_hat)), Y_hat, label="Y_hat")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == "__main__":
    main()
