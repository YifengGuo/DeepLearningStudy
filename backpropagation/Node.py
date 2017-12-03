import math
# class Node is to record and maintain self-information, upstream and downstream connection
# of this node and to implement computation of output aj and error delta


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        constructor of Node
        :param layer_index: the index of layer the node belongs to
        :param node_index: the index of node
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0.0
        self.delta = 0.0

    def set_output(self, output):
        '''
        set node's output, it will be used if this node belongs to input layer
        like x1, x2 ... xn
        :param output: 
        :return: 
        '''
        self.output = output

    def append_upstream_connection(self, conn):
        '''
        append a connection to current node's upstream set
        :param conn: 
        :return: 
        '''
        self.upstream.append(conn)

    def append_downstream_connection(self, conn):
        '''
        append a connection to current node's downstream set
        :param conn: 
        :return: 
        '''
        self.downstream.append(conn)

    def calc_output(self):
        '''
        calculate the output of current node
        :return: 
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
                        self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        if node belongs to hidden layer
        delta_i = ai(1 - ai) * sigma(w_ki * delta_k) k belongs to outputs
        :return: 
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0
        )
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        delta_j = ai(1 - ai) * (label - ai)
        :return: 
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        print node info
        :return: 
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output,
                                                    self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str