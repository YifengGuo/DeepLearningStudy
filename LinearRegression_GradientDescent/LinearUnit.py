from Perceptron import Perceptron
# The only difference between perceptron and linear regression is
# activation function
# The model and train rules are all the same
# so we can directly use perceptron we have implemented already
# and make some changes to make it as linear regression model


# inherit from Percptron to reuse the code
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, lambda x : x)
        # use y = x as activation function
