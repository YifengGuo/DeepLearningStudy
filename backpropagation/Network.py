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
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, dataset, iteration, rate):
        '''
        train the NN
        :param labels: target label of dataset 
        :param dataset: a matrix contain the features of each sample
        :param interation: training times
        :param rate: learning rate
        :return: 
        '''
        for i in range(iteration):
            for data in range(len(dataset)):
                self.train_one_sample(lables[data], data_set[data], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        inner function to train the NN with one sample 
        :param label: 
        :param sample: 
        :param rate: 
        :return: 
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        inner function to compute each node's delta
        :param label: 
        :return: 
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta[label[i]]
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        inner function to update weight of each connection
        :param rate: 
        :return: 
        '''
        for layer in self.layers[:-1]:
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
        self.get_gradient()

    def predict(self, sample):
        '''
        predict the output given the input sample data
        :param sample: 
        :return: 
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        '''
        print info of NN
        :return: 
        '''
        for layer in self.layers:
            layer.dump()
