from Perceptron import Perceptron


def f(x):
    '''
    activator
    :return: 
    '''
    return 1 if x > 0 else 0


def get_training_dataset():
    '''
    based on AND truth table initialize training set
    :return: 
    '''
    # construct training data
    # input list of vector
    # and its corresponding label
    input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    # initialize a perceptron
    p = Perceptron(2, f)
    # set training iteration as 10
    # learning rate as 0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print and_perceptron
    # test
    print '1 and 1 = %d' % and_perceptron.predict([1, 1])
    print '1 and 0 = %d' % and_perceptron.predict([1, 0])
    print '0 and 1 = %d' % and_perceptron.predict([0, 1])
    print '0 and 0 = %d' % and_perceptron.predict([0, 0])