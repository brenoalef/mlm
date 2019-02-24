import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import mnist
from mlm import MLM

class GDMLM:
    def __init__(self, learning_rate=0.01, training_iters=20):
            self.learning_rate = learning_rate
            self.training_iters = training_iters

    def train(self, X, Y, k=None):
        if k == None:
            k = int(round(len(X)*0.4))
        idx = np.random.choice(np.arange(X.shape[0]), k, replace=False)
        self.R = X[idx]
        self.T = Y[idx]
        Dx = euclidean_distances(X, self.R)
        Dy = euclidean_distances(Y, self.T)

        loss_history = []
        self.B_hat = np.zeros((Dx.shape[1], Dy.shape[1]))
        for i in range(self.training_iters):
            error = Dx.dot(self.B_hat) - Dy
            loss = np.sum(np.square(error)) / (k * Dx.shape[0])
            loss_history.append(loss)
            gradient = Dx.T.dot(error)/ Dx.shape[0]
            gradient *= self.learning_rate
            self.B_hat -= gradient
        return loss_history

    def predict(self, X):
        Dx = euclidean_distances(X, self.R)
        Dy = Dx.dot(self.B_hat)
        pred_hat = []
        for i in range(len(X)):
            pred_hat.append(self.T[np.argmin(Dy[i])])
        return np.array(pred_hat)


def main():
    dataset = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')
    dataset = dataset[~np.isnan(dataset).any(axis=1), 1:11]
    dataset[:, 9] = dataset[:, 9] / 2 - 1
    train, test = train_test_split(dataset, test_size=0.33)
    X_train = train[:, 0:9]
    Y_train = train[:, 9]
    X_test = test[:, 0:9]
    Y_test = test[:, 9]
    Y_train = Y_train.reshape((X_train.shape[0], 1))
    Y_test = Y_test.reshape((X_test.shape[0], 1))

    learning_rate = 0.00001
    training_iters = 20
    gdmlm = GDMLM(learning_rate=learning_rate, training_iters=training_iters)
    cost = gdmlm.train(X_train, Y_train, int(round(len(X_train)*0.5)))
    Y_hat = gdmlm.predict(test[:, 0:9])

    plt.plot(range(training_iters), cost)
    plt.show()

    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print(conf_matrix)
    print(1 - np.mean(abs(Y_test - Y_hat)))

    labels = ['Benigno', 'Maligno']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    plt.title('Matriz de Confusão do Classificador')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predito')
    plt.ylabel('Esperado')
    plt.show()

    mlm = MLM()
    mlm.train(X_train, Y_train, int(round(len(X_train)*0.5)))
    Y_hat = mlm.predict(test[:, 0:9])
    
    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print(conf_matrix)
    print(1 - np.mean(abs(Y_test - Y_hat)))


if __name__ == "__main__":
    main()