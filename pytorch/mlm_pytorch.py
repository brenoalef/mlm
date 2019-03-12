import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


class MLM:
    def select_reference_points(self, X, Y, k):
        if k <= 1:
            k = int(round(len(X)*k))
        idx = np.random.choice(np.arange(len(X)), k, replace=False)
        self.R = X[idx]
        self.T = Y[idx]

    def compute_distance(self, U, V):
        '''
        Du = torch.zeros(len(U), len(V))
        for i in range(len(U)):
            print(i)
            for j in range(len(V)):
                Du[i, j] = torch.dist(U[i], V[j])
        return Du
        '''
        return torch.Tensor(euclidean_distances(U.numpy(), V.numpy()))

    def train(self, X, Y, k=0.5):
        self.select_reference_points(X, Y, k)
        Dx = self.compute_distance(X, self.R)
        Dy = self.compute_distance(Y, self.T)
        self.B_hat = torch.pinverse(Dx).mm(Dy)
        return self

    def output_estimation(self, Dy):
        pred_hat = self.T[torch.argmin(Dy, dim=1)]
        return pred_hat

    def predict(self, X):
        Dx = self.compute_distance(X, self.R)
        Dy = Dx.mm(self.B_hat)
        yhat = self.output_estimation(Dy)
        return yhat        


def main():
    digits_dataset = datasets.load_digits()
    X = digits_dataset.data
    Y = digits_dataset.target
    Y = Y.reshape((-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1000, test_size=100)
    Y_train = Y_train.reshape((X_train.shape[0], 1))
    Y_test = Y_test.reshape((X_test.shape[0], 1))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.Tensor(X_train)
    Y_train_tensor = torch.Tensor(Y_train)
    X_test_tensor = torch.Tensor(X_test)
    Y_test_tensor = torch.Tensor(Y_test)

    mlm = MLM()
    mlm.train(X_train_tensor, Y_train_tensor)
    Y_hat_tensor = mlm.predict(X_test_tensor)
    
    Y_hat = Y_hat_tensor.numpy()
    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print(conf_matrix)
    print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    plt.title("Matriz de Confusão do Classificador")
    fig.colorbar(cax)
    ax.set_xticklabels([""] + labels)
    ax.set_yticklabels([""] + labels)
    plt.xlabel("Predito")
    plt.ylabel("Esperado")
    plt.show()

    
if __name__ == "__main__":
    main()
