import numpy as np

def gradient_descent(f_der, states, max_iter = 1000, learning_rate = 0.001, precision = 0.00001):
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

    def f(x, y, z):
        return np.sin(x) + np.cos(y) + np.sin(z)

    def f_der(state):
        x = state[0]
        y = state[1]
        z = state[2]
        return np.array([np.cos(x), -np.sin(y), np.cos(z)])


    state = np.array([1.0, 2.0, 3.5])

    lst, cur = gradient_descent(f_der, state)

    print(cur)


if __name__ == '__main__':
    trial()