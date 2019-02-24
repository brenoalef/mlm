import numpy as np
import matplotlib.pyplot as plt
from mlm import MLM
import mnist
from sklearn.metrics import confusion_matrix

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


    mlm = MLM()
    mlm.train(X_train, Y_train, int(round(len(X_train)*0.5)))
    Y_hat = mlm.predict(X_test)

    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print("Confusion Matrix", conf_matrix)

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
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
