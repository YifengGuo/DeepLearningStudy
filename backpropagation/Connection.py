import random
# the links between nodes of different adjacent layer
# Connection is to record the weight of link and nodes on the each end of link
# from upstream and downstream


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        constructor of Connection
        initial weight is a tiny random number
        initial gradient is 0.0
        :param upstream_node: 
        :param downstream_node: 
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        calculate gradient
        :return: 
        '''
        # from the derivative formula
        # we have delta(Ed) / (delta Wji) = (delta Ed / delta net_j) * Xji
        # and delta Ed / delta net_j is downstream delta named delta_j
        # Xji is input of downstream node and also the output of upstream node
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        '''
        get current gradient
        :return: 
        '''
        return self.gradient

    def update_weight(self, rate):
        '''
        update weight by gradient descent
        :param rate: 
        :return: 
        '''
        # calc by wji = wji + rate * delta_j * Xji
        # and delta_j is downstream node delta
        # Xji is input of downstream node and also the output of upstream node
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        print connection info
        :return: 
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight
        )