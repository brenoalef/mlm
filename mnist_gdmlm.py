import numpy as np
from gdmlm import GDMLM
import matplotlib.pyplot as plt
import mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    learning_rate = 0.0000001
    training_iters = 10
    
    gdmlm = GDMLM(learning_rate=learning_rate, training_iters=training_iters)
    cost = gdmlm.train(X_train, Y_train, int(round(len(X_train)*0.5)))
    Y_hat = gdmlm.predict(X_test)

    print(cost[0])
    print(cost[-1])
    plt.plot(np.arange(0, training_iters), cost)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

    print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

if __name__ == "__main__":
    main()
