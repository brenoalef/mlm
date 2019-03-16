import os
import gzip
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from mlm import MLM

def load_mnist(path, kind='train'):
    

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def main():
    train_images, train_labels = load_mnist('./data/fashion', kind='train')
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images, test_labels = load_mnist('./data/fashion', kind='t10k')
    test_images = test_images.reshape((test_images.shape[0], -1))

    iters = 1
    accuracy = np.zeros((iters, 1))
    models = []
    for i in range(iters):
        train_ids = np.random.choice(len(train_labels), 60000)
        X_train = train_images[train_ids]
        Y_train = train_labels[train_ids]
        Y_train = Y_train.reshape(Y_train.shape[0], 1)
        test_ids = np.random.choice(len(test_labels), 10000)
        X_test = test_images[test_ids]
        Y_test = test_labels[test_ids]
        Y_test = Y_test.reshape(Y_test.shape[0], 1)

        mlm = MLM()
        #mlm.train(X_train, Y_train, k=1000, srp='kmedoids')
        mlm.train(X_train, Y_train, k=2500, srp='rand')
        Y_hat = mlm.predict(X_test)

        accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)
        models.append([mlm, X_test, Y_test, Y_hat])

    mlm, X_test, Y_test, Y_hat = models[(np.abs(accuracy - np.mean(accuracy))).argmin()]

    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print("Confusion Matrix", conf_matrix)
    print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    plt.title("Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xticklabels([""] + labels)
    ax.set_yticklabels([""] + labels)
    plt.xlabel("Predicted")
    plt.ylabel("Desired")
    plt.show()

if __name__ == "__main__":
    main()
