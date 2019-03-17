import numpy as np
import matplotlib.pyplot as plt
from mlm import MLM
from cifar100_web import cifar100
from sklearn.metrics import confusion_matrix

def main():
    
    train_images, train_labels_coarse, train_labels_fine, \
        test_images, test_labels_coarse, test_labels_fine = cifar100(path="./data/cifar100") 
    
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))
    train_labels_coarse = np.argmax(train_labels_coarse, axis=1)
    train_labels_fine = np.argmax(train_labels_fine, axis=1)
    test_labels_coarse = np.argmax(test_labels_coarse, axis=1)
    test_labels_fine = np.argmax(test_labels_fine, axis=1)

    train_labels = np.stack((train_labels_coarse, train_labels_fine), axis=1)
    test_labels = np.stack((test_labels_coarse, test_labels_fine), axis=1)

    iters = 1
    accuracy = np.zeros((iters, 1))
    models = []
    for i in range(iters):
        train_ids = np.random.choice(len(train_labels), 5000)
        X_train = train_images[train_ids]
        Y_train = train_labels[train_ids]
        Y_train = Y_train.reshape(Y_train.shape[0], -1)
        test_ids = np.random.choice(len(test_labels), 1000)
        X_test = test_images[test_ids]
        Y_test = test_labels[test_ids]
        Y_test = Y_test.reshape(Y_test.shape[0], -1)

        mlm = MLM()
        print("Training")
        #mlm.train(X_train, Y_train, 10000, 'kmedoids')
        mlm.train(X_train, Y_train, 500, 'rand')
        print("Testing")
        Y_hat = mlm.predict(X_test)

        accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)
        models.append([mlm, X_test, Y_test, Y_hat])

    mlm, X_test, Y_test, Y_hat = models[(np.abs(accuracy - np.mean(accuracy))).argmin()]

    print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))
    conf_matrix_super_class = confusion_matrix(Y_test[:, 0], Y_hat[:, 0])
    print("Confusion Matrix", conf_matrix_super_class)
    conf_matrix_class = confusion_matrix(Y_test[:, 1], Y_hat[:, 1])
    print("Confusion Matrix", conf_matrix_class)

    


if __name__ == "__main__":
    main()
