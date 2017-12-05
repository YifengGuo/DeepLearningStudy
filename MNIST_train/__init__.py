# -*- coding: UTF-8 -*-
# MNIST has 10,000 test samples and 60,000 train samples
# Determination on hyper-parameters:
# m = sqrt(n + l) + a
# m = log_2n
# m = sqrt(n*l)
# m: # of nodes in hidden layers
# n: # of nodes in input layers
# l: # of nodes in output layers
# a: A constant between 1 and 10
# intuitively for this experiment, we set m = 300
# the output shall be the one out of 10 with largest value (10 classification problem)
# the input of MNIST is 28 * 28 pixel picture (784 pixel)
# and each pixel is corresponding to a node in input layer
# then we have a 784 * 300 * 10 FC neural network
# and so the # of parameters is : (784 + 1) * 300 + 10 * (300 + 1) = 238510 parameters

# error ratio = (wrong prediction samples) / total samples

# so to process input data to the form NN can accept:
# we convert the 28 * 28 pic into a 784 dimension vector
# each tag is a value between 0 and 9
# then we convert it to a 10-dimensional one-hot vector
# for example: if the value of tag is n, we st nth dimension to 0.9, the others is 0.1
# [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] represents 2

# In natural language processing, a one-hot vector is a 1 × N matrix (vector) used to distinguish
# each word in a vocabulary from every other word in the vocabulary. The vector consists of 0s in all
# cells with the exception of a single 1 in a cell used uniquely to identify the word.
import struct
from backpropagation import *
from datetime import datetime


# to load the train data
class Loader(object):
    def __init__(self, path, count):
        '''
        constructor of Loader
        :param path: data path
        :param count: the number of samples in the file
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        read the data from file
        :return: 
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        '''
        convert unsigned byte to integer
        :param byte: 
        :return: 
        '''
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        get pictures from the file
        :param content: 
        :param index: 
        :return: one picture
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start + 1 * 28 + j])
                )
        return picture

    def get_one_sample(self, picture):
        '''
        convert picture into the form of input vector for NN
        :param picture: 
        :return: vector of one sample
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        load data, obtain the input vector of all samples
        :return: data set
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)
                )
            )
        return data_set


class LabelLoader(Loader):
    def load(self):
        '''
        load data, obtain the label vector of all samples
        :return: vector of labels
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        normalize a label into 10-dimensional vector 
        :param label: 
        :return: 
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    '''
    obtain training data_set
    :return: 
    '''
    image_loader = ImageLoader('train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_t_data_set():
    '''
    obtain test data_set
    :return: 
    '''
    image_loader = ImageLoader('t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    '''
    the output of NN is a 10-dimensional vector (0 - 9)
    what we need is the ith tag index with largest value
    :param vec: 
    :return: 
    '''
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    '''
    error ratio evaluation
    :param network: NN object
    :param test_data_set: test data set offered by MNIST
    :param test_labels: test lables offered by MNIST
    :return: 
    '''
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)  # error ratio = wrong predictions / total test samples


def train_and_evaluate():
    '''
    strategy for training and evaluating the network
    :return: 
    '''
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_t_data_set()
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(test_labels, train_data_set, 0.3, 1)
        print '%s epoch %d finished ' % (datetime.now(), epoch)

        if epoch % 10 == 0: # evaluate error ration every 10 iterations
            error_ratio = evaluate(network, test_data_set, test_labels)
            print '%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()




