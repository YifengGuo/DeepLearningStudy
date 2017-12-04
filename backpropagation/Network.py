# offer APIs to the external world
from Connection import Connection
from Connections import Connections
from Layer import Layer


class Network(object):
    def __init__(self, layers):
        '''
        initialize a FC neural network
        :param layers: A two-dimension array, which is to record the number of nodes in each layer
        len(layers) = number of layer in NN
        len(layers[i]) = the number of nodes in ith layer 
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers) # number of layer
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i])) # Layer[i] is the number of nodes in ith layer
        for layer in range(layer_count - 1):  # one connection between two layers
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes  # upstream_node at this iteration
                           for downstream_node in self.layers[layer + 1].nodes[:-1]] # downstream_node at this iteration
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, dataset, rate, iteration):
        '''
        train the NN
        :param labels: target label of dataset 
        :param dataset: a matrix contain the features of each sample
        :param rate: learning rate
        :param iteration: training times
        :return: 
        '''
        for i in range(iteration):
            for data in range(len(dataset)):
                self.train_one_sample(labels[data], dataset[data], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        inner function to train the NN with one sample 
        :param label: target label of one data vector of features
        :param sample: a matrix of data, each line is list of features of one data
        :param rate: learning rate
        :return: 
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        inner function to compute each node's delta
        using back propagation algorithm
        :param label: 
        :return: 
        '''
        # nodes in output layers
        output_nodes = self.layers[-1].nodes
        # calculate delta of output layer
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta[label[i]]
        # calculate delta of hidden layers
        for layer in self.layers[-2::-1]:  # from second last to the first layer
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        inner function to update weight of each connection
        :param rate: 
        :return: 
        '''
        for layer in self.layers[:-1]:  # except last(output) layer
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        inner function to compute the gradient of each connection
        :return: 
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        obtain the gradient of each connection given a certain label of NN
        :param label: 
        :param sample: 
        :return: 
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        predict the output given the input sample data
        the result will be used to calculate delta and then update weight
        :param sample: 
        :return: 
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        # return the list of output (prediction) of output layer except ConstNode
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])  # the last node is ConstNode

    def dump(self):
        '''
        print info of NN
        :return: 
        '''
        for layer in self.layers:
            layer.dump()
