import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, Y, f_der, states, max_iter = 1000, learning_rate = 0.001, precision = 0.00001):

    cur = np.array(states)
    last = np.array([float('inf'), float('inf')])
    lst = [cur]


    for _ in range(max_iter):

        last = cur.copy()
        m = cur[0]
        c = cur[1]
        gradient = f_der(X, Y, m , c)

        cur = (cur - gradient * learning_rate)

        if np.all(abs(cur - last)) < precision :
            break

        lst.append(cur)

    return lst, cur




def trial():

    x = np.linspace(1, 50, 50)
    y = x + 5

    def func(x, y, m, c):
        return np.sum(((m * x + c) - y) ** 2) / (2 * len(y))

    def func_der(x, y, m, c):
        error = m * x + c - y

        dm = np.sum(error * x) / len(y)
        dc = np.sum(error) / len(y)

        return np.array([dm, dc])


    x = np.linspace(1, 50, 50)
    y = x + 5

    init_m, init_c = 5, 5
    states = np.array([init_m,init_c])

    lst, bestWieghts = gradient_descent(x, y,func_der, states)

    m = bestWieghts[0]
    c = bestWieghts[1]

    plt.plot(x, y, '--r')
    plt.scatter(x, m * x + c)
    plt.show()


if __name__ == '__main__':
    trial()