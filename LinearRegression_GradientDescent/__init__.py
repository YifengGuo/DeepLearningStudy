from LinearRegression_GradientDescent.LinearUnit import LinearUnit


def get_training_dataset():
    '''
    fabricate 5 people data
    input_vec has one feature which represents working years
    label represents the salary the person should have
    :return: 
    '''
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    lu = LinearUnit(1) # one input feature
    input_vecs, labels = get_training_dataset()
    # set input vectors, labels (data matrix) and
    # train iterations and learning rate to train
    # the linear regression model
    lu.train(input_vecs, labels, 10, 0.01)
    return lu


if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print linear_unit

    # test
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])



