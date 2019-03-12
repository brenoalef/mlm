import torch
import numpy as np
import matplotlib.pyplot as plt

import mnist
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlm_pytorch import MLM


def main():
    train_images = mnist.train_images()
    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
    test_labels = mnist.test_labels()

    train_ids = np.random.choice(len(train_labels), 5000)
    X_train = train_images[train_ids]
    Y_train = train_labels[train_ids]
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    test_ids = np.random.choice(len(test_labels), 100)
    X_test = test_images[test_ids]
    Y_test = test_labels[test_ids]
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

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
