import torch                                        # root package
import torch.autograd as autograd                   # computation graph
from torch.autograd import Variable                 # variable node in computation graph
import torch.nn as nn                               # neural networks
import torch.nn.functional as F                     # layers, activations and more
import torch.optim as optim                         # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace                 # hybrid frontend decorator and tracing jit

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LinearReg(nn.Module):
    def  __init__(self, input_dim, output_dim):
        super(LinearReg, self).__init__() 
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


class SGDMLM:
    def __init__(self, training_iters=20, learning_rate=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False, scheduler_step_size=40, scheduler_gamma=0.1):
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

    def select_reference_points(self, X, Y, k):
        idx = np.random.choice(np.arange(len(X)), k, replace=False)
        self.R = X[idx]
        self.T = Y[idx]

    def compute_distance(self, U, V):
        Du = torch.zeros(len(U), len(V))
        for i in range(len(U)):
            for j in range(len(V)):
                Du[i, j] = torch.dist(U[i], V[j])
        return Du

    def train(self, X, Y, k=0.5):
        if k <= 1:
            k = int(round(len(X)*k))
        '''
        X, idx = np.unique(X, axis=0, return_index=True)
        Y = Y[idx]
        '''
        #X = torch.unique(X, dim=0)
        self.select_reference_points(X, Y, k)
        Dx = self.compute_distance(X, self.R)
        Dy = self.compute_distance(Y, self.T)
        dx_data = Variable(Dx)
        dy_data = Variable(Dy)
        linearReg = LinearReg(k, k)
        criterion = nn.MSELoss(size_average = False) 
        optimizer = optim.SGD(linearReg.parameters(), lr = self.learning_rate, momentum = self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        loss_history = []
        for epoch in range(self.training_iters):
            scheduler.step()
            r = torch.randperm(len(dx_data))
            epoch_loss = 0.0
            for i in range(len(dx_data)):
                pred_dy = linearReg(dx_data[r[i]]) 
                loss = criterion(pred_dy, dy_data[r[i]]) 
                epoch_loss += loss.item()
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()
            loss_history.append(epoch_loss / len(dx_data))
            print('epoch {}, loss {}'.format(epoch + 1, loss_history[epoch]))
        self.loss_history = loss_history
        self.linearReg = linearReg
        return self

    def output_estimation(self, Dy):
        pred_hat = self.T[torch.argmin(Dy, dim=1)]
        return pred_hat

    def predict(self, X):
        Dx = self.compute_distance(X, self.R)
        Dy = self.linearReg(Variable(Dx)).data
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

    training_iters = 50
    sgdmlm = SGDMLM(learning_rate=9.0e-6, training_iters=training_iters)
    sgdmlm = sgdmlm.train(X_train_tensor, Y_train_tensor)
    Y_hat_tensor = sgdmlm.predict(X_test_tensor)
    
    plt.plot(range(1, training_iters + 1), sgdmlm.loss_history, "-o")
    plt.show()

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
