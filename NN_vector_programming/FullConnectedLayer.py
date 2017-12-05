# re-form the layer implementation in backpropagation
# this new class implements forward and backward calculation in full connected layers
import numpy as np


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        constructor of class
        :param input_size: the dimensions of input vector to current layer
        :param output_size: the dimensions of output vector from current layer
        :param activator: activation function (sigmoid function) 
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # vector of weights W
        self.W = np.random.uniform(
            -0.1, 0.1,
            (output_size, input_size)
        )

        # bias term b
        self.b = np.zeros((output_size, 1))

        # vector of output
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        farward calculation
        :param input_array: vector of input, its dimension must be equal to input_size 
        :return: 
        '''
        # a sigmoid(w * x) in vector form
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b
        )

    def backward(self, delta_array):
        '''
        backpropagation to calculate gradient of W and b
        :param delta_array: the vector delta_j calculated and sent from the next layer
        :return: 
        '''
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array
        )
        self.W_gradient = np.dot(delta_array, self.input.T)
        self.b_gradient = delta_array

    def update(self,learning_rate):
        '''
        update weight by gradient descent
        :param learning_rate: 
        :return: 
        '''
        self.W += learning_rate * self.W_gradient
        self.b += learning_rate * self.b_gradient

    