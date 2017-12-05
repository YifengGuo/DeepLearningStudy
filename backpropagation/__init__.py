from Network import Network
from Normalizer import Normalizer
from Connection import Connection
from Connections import Connections
import random
# As shown in model image, the implementation of neural network can be
# accomplished by 5 parts with some relationships

# Network: the object of neural networks, it offers certain APIs. And it is
#          composed of several Layer objects and Connections objects

# Layer: it is composed of several Node objects

# Node: it is to compute and record the info of neuron(like output a, error delta)
#       as well as associated downstream and upstream connection of this node

# Connection: each connected object shall record the weight of this connection (e.g W42)

# Connections: initialized as the set of Connection and support the operations of set


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(
        lambda a, b: a + b,
        map(lambda v: (v[0] - v[1]) * (v[0] - v[1])),
        zip(vec1, vec2)
    )


def gradient_check(network, sample_feature, sample_label):
    '''
    gradient check
    :param network: NN object
    :param sample_feature:
    :param sample_label:
    :return:
    '''
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # obtain gradient of each connection of this sample
    network.get_gradient(sample_feature, sample_label)

    # for each gradient, do the gradient check
    for conn in network.connections.connections:
        # obtain the gradient of current connection
        actual_gradient = conn.get_gradient()

        # add a very small value to the weight and re-calculate error
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # substract a very small value to the weight and re-calculate error
        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        # based on the definition of derivative
        # delta_f(wji) / delta_wji = (f(wji + epsilon) - (wji - epsilon)) / (2 * epsilon)
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # print
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient,
            actual_gradient
        )


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def t(network, data):
    '''
    test the model with input data
    :param network: 
    :param data: 
    :return: 
    '''
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print '\ttestdata(%u)\tpredict(%u)' % (
        data,
        normalizer.denorm(predict_data)
    )


def correct_ratio(network):
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)

