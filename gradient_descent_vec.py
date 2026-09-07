import numpy as np

def gradient_descent_vec(f_der, states, max_iter = 1000,learning_rate = 0.001, precision = 0.0001):
    cur = np.array(states)
    last = cur + float('inf')

    lst = [cur]

    for _ in range(max_iter):
        last = cur.copy()
        gradient = f_der(cur)
        cur = (cur - gradient * learning_rate)

        if np.all(abs(cur - last)) < precision:
            break

        lst.append(cur)


    return lst, cur


def trial():

    def f(x, y):
        return (2 * x ** 2) - (4 * x * y) + (y ** 4) + 2

    def f_der_x(x, y):
        return 4 * (x - y)

    def f_der_y(x, y):
        return -4 * x + (4 * y ** 3)

    def vecOfDer(state):
        x, y = state[0], state[1]
        return np.array([f_der_x(x, y), f_der_y(x, y)])


    inital_x, inital_y = 2.5, 1.9
    state = np.array([inital_x, inital_y])

    best , cur = gradient_descent_vec(vecOfDer, state)

    print(f"Best is at {cur}")


if __name__ == '__main__' :
    trial()