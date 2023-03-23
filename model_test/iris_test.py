from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import random, shuffle, seed
import numpy as np
import torch
import bsi_ops
import time

def load_iris_and_train_model(dot_product_function=torch.dot):
    # reset seed to repeat everything that happens
    seed(40)

    iris = load_iris()
    X = [[x[2], x[3]] for x in iris.data]
    y = iris.target.copy()

    for i in range(len(y)):
        if y[i] == 0: y[i] = 1
        else: y[i] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    dim = len(X[0])
    w = torch.autograd.Variable(torch.rand(dim), requires_grad=True)
    b = torch.autograd.Variable(torch.rand(1), requires_grad=True)

    step_size = 1e-3
    num_epochs = 5000
    minibatch_size = 20

    start = time.time()
    # print(f'Starting training at epoch {start}')
    
    for _ in range(num_epochs):
        inds = [i for i in range(len(X_train))]
        shuffle(inds)

        for i in range(len(inds)):
            L = max(0, 1 - y_train[inds[i]] * (dot_product_function(w, torch.Tensor(X_train[inds[i]])) -b)) ** 2
            if L != 0:
                L.backward()
                w.data -= step_size * w.grad.data
                b.data -= step_size * b.grad.data
                w.grad.data.zero_()
                b.grad.data.zero_()

    end = time.time()
    print(f'Time taken for training: {end - start}')

    def accuracy(X, y):
        correct = 0
        for i in range(len(y)):
            y_pred = int(np.sign((torch.dot(w, torch.Tensor(X[i])) -b).detach().numpy()[0]))
            if y_pred == y[i]: correct += 1

        return float(correct) / len(y)
    
    print('plane equation:  w=', w.detach().numpy(), 'b =', b.detach().numpy()[0])
    print('train accuracy:', accuracy(X_train, y_train))
    print('test accuracy:', accuracy(X_test, y_test))


if __name__ == '__main__':
    # run with torch.dot function
    load_iris_and_train_model()
    load_iris_and_train_model()
    load_iris_and_train_model(dot_product_function=bsi_ops.dot_product)
