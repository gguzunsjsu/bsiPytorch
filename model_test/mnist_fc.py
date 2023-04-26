import numpy as np
import torch
import bsi_ops

from personal_net.network import Network
from personal_net.layers import FCLayer, ActivationLayer
from personal_net.activations import tanh, tanh_prime
from personal_net.losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

import time

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# convert everything to a tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def train_and_time_network(dot_product_function=torch.dot):
    start = time.time()

    # Network
    net = Network()
    net.add(FCLayer(28*28, 100, dot_function=dot_product_function))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 50, dot_function=dot_product_function))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(50, 10, dot_function=dot_product_function))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
    net.add(ActivationLayer(tanh, tanh_prime))

    # train on 1000 samples
    # as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
    net.use(mse, mse_prime)
    net.fit(x_train[0:500], y_train[0:500], epochs=35, learning_rate=0.1)

    # test on 3 samples
    out = net.predict(x_test[0:3])
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(y_test[0:3])

    end = time.time()

    return end - start

def main():
    dot_product_functions = [torch.dot, bsi_ops.dot_product]
    dot_product_function_names = ['torch.dot', 'bsi_osp.dot_product']

    for func, func_name in zip(dot_product_functions, dot_product_function_names):
        time_taken = train_and_time_network(func)

        print(f"{func_name} took {time_taken} amount of time")


main()