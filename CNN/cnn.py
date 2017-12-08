import numpy as np
from Activators.activators import *

def get_patch(input_array, i, j, filter_width, filter_height, stride):
    '''
    get the area of current convolution from input array
    automatically match 2 dimensional or 3 dimensional input cases
    :param input_array: 
    :param i: 
    :param j: 
    :param filter_width: 
    :param filter_height: 
    :param stride: 
    :return: 
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
                start_i : start_i + filter_height,
                start_j: start_j + filter_width
        ]
    elif input_array.ndim == 3:
        return input_array[:,
            start_i : start_i + filter_height,
            start_j : start_j + filter_width

        ]

# obtain the index of element with max value of 2d area
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


def conv(input_array, kernel_array,
         output_array, stride, bias):
    '''
    compute convolution
    automatically match 2 dimensional or 3 dimensional cases
    :param input_array: 
    :param kernel_array: 
    :param output_array: 
    :param stride: 
    :param bias: 
    :return: 
    '''
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width, kernel_height, stride) \
                * kernel_array).sum() + bias

# add zeros padding to an array
def padding(input_array, zp):
    '''
    add zeros padding to an array
    automatically match 2 dimensional or 3 dimensional cases
    :param input_array: 
    :param zp: 
    :return: 
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp
            ))
            padded_array[:,
            zp : zp + input_height,
            zp : zp + input_width
            ] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp
            ))
            padded_array[
                zp : zp + input_height,
                zp: zp + input_width
            ] = input_array
            return padded_array


# do element wise operation on numpy array
def element_wise_op(array, op):
    for i in np.nditer(
        array, op_flags=['readwrite']
    ):i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape
        )
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights: \n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias)
        )

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad



class ConvLayer:
    def __init__(self, input_width, input_height,
                 channel_number, filter_width, filter_height,
                 filter_number, zero_padding, stride, activator,
                 learning_rate):
        '''
        constructor of Convolution Layer
        :param input_width: 
        :param input_height: 
        :param channel_number: 
        :param filter_width: 
        :param filter_height: 
        :param filter_number: 
        :param zero_padding: 
        :param stride: 
        :param activator: 
        :param learning_rate: 
        '''
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        # the output of convolution layer is determined by input_width, filter_width, zero_padding and stride
        self.output_width = \
            ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding,
            stride)
        self.output_height = \
            ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding,
            stride)
        self.output_array = np.zeros((self.filter_number,
            self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        '''
        calculate the output of convolution layer
        and store the result in the output_array
        :param input_array: 
        :return: 
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        '''
        calculate the delta term of last layer, and calculate the 
        gradient of each weight.
        the delta terms of last layer are stored in the self.delta_array
        gradients are stored in the weight_grad field of class Filter
        :param input_array: 
        :param sensitivity_array: 
        :param activator: 
        :return: 
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        '''
        update weights by gradient descent
        :return: 
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        '''
        calculate the sensitivity map (containing delta terms) which
        needs transmitting to last layer
        :param sensitivity_array: sensitivity map of current layer
        :param activator: the activator of last layer
        :return: 
        '''
        # deal with stride of convolution, expand the original sensitivity_map
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # full convolution, zero pad the sensitivity map
        expanded_width = expanded_array.shape[2]
        zp = (
            self.input_width + self.filter_width - 1 - expanded_width
        ) / 2
        padded_array = padding(expanded_array, zp)
        # initialize delta_array to store the sensitivity map which
        # needs transmitting to the last layer
        self.delta_array = self.create_delta_array()
        # for the convolution layers which have multiple filters, the sensitivity map
        # which is transmitted to the last layer, is the sum of sensitivity maps of all
        # filters
        for f in range(self.filter_number):
            filter = self.filters[f]
            # filp the weight map by 180 degrees
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()
            ))
            # calculate the corresponding delta_array of each filter
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        # do element-wise operation of the result and activator function
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # deal with conv stride, expand the original sensitivity map
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # calculate gradient of each weight
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # calculate the gradient of bias term

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # determine the size of expanded array
        # calculate the size of sensitivity map when stride = 1
        expanded_width = self.input_width - self.filter_width + 2 * self.zero_padding + 1
        expanded_height = self.input_height - self.filter_height + 2 * self.zero_padding + 1
        # construct new sensitivity map
        expanded_array = np.zeros((depth, expanded_height, expanded_width))
        # copy delta from original sensitivity map
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expanded_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expanded_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        '''
        determine the size of output of convolution layer
        :param input_size: 
        :param filter_size: 
        :param zero_padding: 
        :param stride: 
        :return: 
        '''
        return (input_size - filter_size + 2 * zero_padding) / stride + 1

# implementation on max pooling layer
class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width -
                             filter_width) / self.stride + 1
        self.output_height = (input_height -
                              filter_height) / self.stride + 1
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max()
                    )

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride
                    )
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                    i * self.stride + k,
                    j * self.stride + l] = \
                    sensitivity_array[d, i, j]


def init_t():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def t():
    a, b, cl = init_t()
    cl.forward(a)
    print cl.output_array


def t_bp():
    a, b, cl = init_t()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print cl.filters[0]
    print cl.filters[1]


def gradient_check():
    '''
    gradient check
    '''
    # define a loss function which is to sum the result of neurons
    error_function = lambda o: o.sum()

    # forward calculation
    a, b, cl = init_t()
    cl.forward(a)

    # get sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape,
                                dtype=np.float64)
    # compute gradient
    cl.backward(a, sensitivity_array,
                IdentityActivator())
    # check gradient
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print 'weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j])


def init_pool_t():
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2],
          [2, 4]],
         [[3, 5],
          [8, 2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl


def t_pool():
    a, b, mpl = init_pool_t()
    mpl.forward(a)
    print 'input array:\n%s\noutput array:\n%s' % (a,
                                                   mpl.output_array)


def t_pool_bp():
    a, b, mpl = init_pool_t()
    mpl.backward(a, b)
    print 'input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
        a, b, mpl.delta_array)


if __name__ == '__main__':
    init_t()
    t()
    t_bp()
    gradient_check()
    init_pool_t()
    t_pool()
    t_pool_bp()