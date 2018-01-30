import numpy as np
def calc_standard_deviation(input):
    '''
    
    :param input: 1D input data
    :return: standard deviation of input
             
             s = sqrt(Singma_1_n((x_i - x_avg) ^ 2) / (n - 1))
    '''
    avg = calc_average(input)
    total = 0
    for item in input:
        total += (item - avg) ** 2
    return np.sqrt(total / (len(input) - 1))


def calc_average(input):
    return np.mean(input)


if __name__ == '__main__':
    input = [10,2,38,23,38,23,21]
    print calc_standard_deviation(input)