import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlm import MLM
from sgdmlm import SGDMLM

boston_dataset = datasets.load_boston()
X = boston_dataset.data
Y = boston_dataset.target
Y = Y.reshape((-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=337, test_size=169)
Y_train = Y_train.reshape((X_train.shape[0], 1))
Y_test = Y_test.reshape((X_test.shape[0], 1))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
learning_rate = 0.00001
training_iters = 20
sgdmlm = SGDMLM(learning_rate=learning_rate, training_iters=training_iters)
cost = sgdmlm.train(X_train, Y_train, int(round(len(X_train)*0.4)))
Y_hat = sgdmlm.predict(X_test)

plt.plot(range(training_iters), cost)
plt.show()

mse = (np.square(Y_test - Y_hat)).mean(axis=0)
print(mse)
'''


mlm = MLM()
mlm.train(X_train, Y_train, k=0.2)
Y_hat = mlm.predict(X_test, method="nn")

error = Y_test - Y_hat
mse = np.square(error).mean(axis=0)
print("MSE:", mse)
print("RMSE:", mse ** (1/2))

nmse = np.mean(np.square(error)/(np.mean(Y_test) * np.mean(Y_hat)), axis=0)
print("NMSE: ", nmse)
