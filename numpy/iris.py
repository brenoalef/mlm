import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlm import MLM
from sgdmlm import SGDMLM

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target
Y = Y.reshape((-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=100, test_size=50)
Y_train = Y_train.reshape((X_train.shape[0], 1))
Y_test = Y_test.reshape((X_test.shape[0], 1))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

learning_rate = 0.001
training_iters = 20
sgdmlm = SGDMLM(learning_rate=learning_rate, training_iters=training_iters)
cost = sgdmlm.train(X_train, Y_train, len(X_train))
Y_hat = sgdmlm.predict(X_test)


plt.plot(range(training_iters), cost)
plt.show()


conf_matrix = confusion_matrix(Y_test, Y_hat)
print(conf_matrix)
print(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))

labels = ["Setosa", "Versicolor", "Virginica"]
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

labels = ["Setosa", "Versicolor", "Virginica"]
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

from mlm import model_selection, get_accuracy
k_values = np.arange(0.1,1.0,0.1)
k = model_selection(X_train, Y_train, k_values, 10)
model = MLM()
model.train(X_train, Y_train, k=k)
pred = model.predict(X_test)
print("Koptim:", k)
print("Accuracy:", get_accuracy(Y_test, pred))