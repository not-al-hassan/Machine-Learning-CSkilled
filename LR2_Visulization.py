
# Homework 1 --> Implementation of Linear Regression using Gradient Descent.

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y, states ,  f_der, max_iter = 1000, learning_rate = 0.1, precision = 0.00001):

    cur = np.array(states)
    last = cur + 100 + precision
    list = [cur.copy()]

    for _ in range(max_iter):
        last = cur.copy()
        gradient = f_der(x, y, cur)
        cur -= (gradient * learning_rate)

        if np.all(abs(cur - last)) < precision:
            break

        list.append(cur)

    return list, cur


def MSE(x, y, weights):
    prediction = x @ weights
    error = prediction - y
    return (error.T @ error) / (2 * len (y))


def MSE_der(x, y, weights):
    prediction = x @ weights
    error = prediction - y
    gradient = x.T @ error / (len (y))

    return gradient




if __name__ == '__main__':

    x = np.array([0, 0.2, 0.4, 0.8, 1.0])
    y = x + 5


    X = np.column_stack((np.ones(len(x)), x))

    states = np.random.rand(2)


    list, ret = gradient_descent(
        X, y, states, MSE_der
    )

    c , m = ret[0], ret[1]

    plt.plot(x, y, 'r')
    plt.scatter(x, m * x + c)
    plt.show()