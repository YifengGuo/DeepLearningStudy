#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Perceptron:
# 1. input vector and each input's weight
#    w0 as bias for the fixed input 1
# 2. Activation function
#    here we use step function
# 3. output: f(x) = w * x + b

# perceptron training:
# set weights and bias as 0 initially
# wi <- wi + delta_wi
# b <- b + delta_b
# in particular:
# delta_wi = ita(t - y)*xi
# delta_b = ita(t - y)
# ita is learning rate

# here can I think perceptron updating weights in a little bit
# brute way. w_new = w_old + delta_w and delta_w = ita * error * x
# so if error is large, change from w_old to w_new is large
#    if x (or importance) is large, change from w_old to w_new is also large

class Perceptron(object):

    def __init__(self, input_num, activator):
        self.activator = activator
        # y = w * x + b, so w and x are equal numerically
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        '''
        print result in string
        :return: 
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''

        :param input_vec: input vecor
        :return: result of prediction
        '''
        # output: f(x) = w * x + b
        # input_vec: [x1, x2, x3...]
        # weights: [w1, w2, w3...]
        # package input_vec with weights -> [(x1, w1), (x2, w2)...(xn, wn)]
        # use map() to calculate [x1*w1, x2*w2, x3*w3...xn*wn]
        # use reduce() to sum up x1*w1 ... xn*wn
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vec, self.weights))
                   , 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        input training data: a list of vector
                             a target label for each input vector
                             training iteration epochs
                             learning rate
        :param input_vecs: 
        :param labels: 
        :param iteration: 
        :param rate: 
        :return: 
        '''

        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        A single iteration to train all the training data for one time
        :param input_vecs: 
        :param labels: 
        :param rate: 
        :return: 
        '''
        # package input vectors with target labels
        # and each training set is (input_vec, label)
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            # calculate each output of training set
            output = self.predict(input_vec)
            # update each vector's weight
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        update weights by rules of perceptron
        :param input_vec: 
        :param output: 
        :param label: 
        :param rate: 
        :return: 
        '''
        # wi <- wi + delta_wi
        # b <- b + delta_b
        # delta_wi = ita(t - y)*xi
        # delta_b = ita(t - y)
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights)
        )

        self.bias = self.bias + rate * delta