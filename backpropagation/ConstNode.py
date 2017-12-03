# to compute bias term like w1x1 + w2x2 + .. + w1b
# we need a constant node to count bias wb


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        constructor of ConstNode
        :param layer_index: the index of layer the node belongs to
        :param node_index: the index of current constnode
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        # self.delta = 0.0

    def append_downstream_connection(self, conn):
        '''
        append a connection to downstream node
        :param conn: 
        :return: 
        '''
        self.downstream.append(conn)

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

    def __str__(self):
        '''
        print info of node
        :return: 
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

