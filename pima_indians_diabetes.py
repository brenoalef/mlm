import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlm import MLM
from sgdmlm import SGDMLM


dataset = np.genfromtxt('pima-indians-diabetes.data', delimiter=',')
#dataset = dataset[~np.isnan(dataset).any(axis=1), 1:11]
train, test = train_test_split(dataset, train_size=512, test_size=256)
X_train = train[:, 0:8]
Y_train = train[:, 8]
X_test = test[:, 0:8]
Y_test = test[:, 8]
Y_train = Y_train.reshape((-1, 1))
Y_test = Y_test.reshape((-1, 1))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

learning_rate = 0.00001
training_iters = 20
sgdmlm = SGDMLM(learning_rate=learning_rate, training_iters=training_iters)
cost = sgdmlm.train(X_train, Y_train, len(X_train))
Y_hat = sgdmlm.predict(X_test)


plt.plot(range(training_iters), cost)
plt.show()


conf_matrix = confusion_matrix(Y_test, Y_hat)
print(conf_matrix)
print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

labels = ["Terá diabetes", "Não terá diabetes"]
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

mlm = MLM()
mlm.train(X_train, Y_train, len(X_train))
Y_hat = mlm.predict(X_test)

conf_matrix = confusion_matrix(Y_test, Y_hat)
print(conf_matrix)
print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

labels = ["Terá diabetes", "Não terá diabetes"]
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
