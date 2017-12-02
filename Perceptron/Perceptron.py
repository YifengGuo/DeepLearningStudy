#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Perceptron:
# 1. input vector and each input's weight
#    w0 as bias for the fixed input 1
# 2. Activation function
#    here we use step function
# 3. output: f(x) = w * x + b

# perceptro training:
# set weights and bias as 0 initially
# wi <- wi + delta_wi
# b <- b + delta_b
# in particular:
# delta_wi = ita(t - y)*xi
# delta_b = ita(t - y)
# ita is learning rate


class Perceptron(object):

    def __init__(self, input_num, activator):
        self.activator = activator
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

        # input_vec: [x1, x2, x3...]
        # weights: [w1, w2, w3...]
        # package input_vec with weights -> [(x1, w1), (x2, w2)...(xn, wn)]
        # use map() to calculate [x1*w1, x2*w2, x3*w3...xn*wn]
        # use reduce() to sum up x0*w0 + x1*w1 ... xn*wn
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
        A single iteration to pass all the training data for one time
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
            lambda (w, x): w + rate * delta * x,
            zip(input_vec, self.weights)
        )

        self.bias = self.bias + rate * delta

    def f(x):
        '''
        activator
        :return: 
        '''
        return 1 if x > 0 else 0

    def get_training_dataset():
        '''
        based on AND truth table training set
        :return: 
        '''
        # construct training data
        # input list of vector
        input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        labels = [1, 0, 0, 0]
        return input_vecs, labels

    def train_and_perceptron():
        # initialize a perceptron
        p = Perceptron(2, f)
        # set training iteration as 10
        # learning rate as 0.1
        input_vecs, labels = get_training_dataset()
        p.train(input_vecs, labels, 10, 0.1)
        return p

    if __name__ == '__main__':
        and_perceptron = train_and_perceptron()
        print and_perceptron
        print '1 and 1 = %d' % and_perceptron.predict([1, 1])
        print '0 and 0 = %d' % and_perceptron.predict([0, 0])
        print '1 and 0 = %d' % and_perceptron.predict([1, 0])
        print '0 and 1 = %d' % and_perceptron.predict([0, 1])





