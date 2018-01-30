from PCA.calc_standard_deviation import *


def calc_covariance(a, b):
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum / (len(a) - 1)

if __name__ == '__main__':
    a = [9, 15, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20]
    b = [39, 56, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80]
    print calc_covariance(a, b)