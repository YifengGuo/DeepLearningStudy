from Node import Node
from ConstNode import ConstNode
# Layer object is to initialize a layer in neural networks.
# Besides, as the set of Node, it also support set operations on Node


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.node_count = node_count
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        set the output of whole layer
        if the current layer is input layer, 
        this function will be used
        :param data: 
        :return: 
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        calculate the output of whole layer
        :return: 
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        print info of whole layer
        :return: 
        '''
        for node in self.nodes:
            print node
