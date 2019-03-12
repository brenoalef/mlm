import numpy as np
import matplotlib.pyplot as plt
from mlm import MLM
import cifar10_web
from sklearn.metrics import confusion_matrix

def main():
    train_images, train_labels, test_images, test_labels = cifar10_web.cifar10(path=None) 
    
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))
    train_labels = np.argmax(train_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)

    iters = 1
    accuracy = np.zeros((iters, 1))
    models = []
    for i in range(iters):
        train_ids = np.random.choice(len(train_labels), 5000)
        X_train = train_images[train_ids]
        Y_train = train_labels[train_ids]
        Y_train = Y_train.reshape(Y_train.shape[0], 1)
        test_ids = np.random.choice(len(test_labels), 100)
        X_test = test_images[test_ids]
        Y_test = test_labels[test_ids]
        Y_test = Y_test.reshape(Y_test.shape[0], 1)

        mlm = MLM()
        mlm.train(X_train, Y_train, int(round(len(X_train)*0.8)))
        Y_hat = mlm.predict(X_test)

        accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)
        models.append([mlm, X_test, Y_test, Y_hat])

    mlm, X_test, Y_test, Y_hat = models[(np.abs(accuracy - np.mean(accuracy))).argmin()]

    conf_matrix = confusion_matrix(Y_test, Y_hat)
    print("Confusion Matrix", conf_matrix)
    print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
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
