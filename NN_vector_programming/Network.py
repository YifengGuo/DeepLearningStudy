import numpy as np
from FullConnectedLayer import FullConnectedLayer


# activator class by sigmoid
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(weighted_input))

    def backward(self, output):
        return output * (1 - output)


class Network(object):
    def __init__(self, layers):
        '''
        constructor
        :param layers: 
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(layers[i], layers[i+1], SigmoidActivator())
            )

    def predict(self, sample):
        '''
        prediction of NN
        :param sample: input sample
        :return: 
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        train function
        :param labels: sample label 
        :param data_set: train data_set
        :param rate: learning_rate
        :param epoch: train iteration
        :return: 
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].\
            activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)