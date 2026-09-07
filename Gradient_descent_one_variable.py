import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(start_x,  func_der, learning_rate, max_iter = 5000, precision = 1e-6):
    cur_x = start_x
    last_x = float('inf')
    lst = [start_x]

    for _ in range(max_iter):
        last_x = cur_x
        gradient = func_der(cur_x)
        cur_x = cur_x - gradient  * learning_rate
        lst.append(cur_x)
        if abs(cur_x - last_x) < precision:
            break

    return lst,cur_x




def func1():

    rangeStart = -10
    rangeEnd = 10

    start_x = -7.5

    def func(x):
        return 3 * x ** 2 + 4 * x + 7

    def funcDer(x):
        return 6 * x + 4

    x = np.linspace(rangeStart, rangeEnd, 50)
    y = func(x)

    gradList, best_x = gradientDescent(
        start_x,
        funcDer,
        0.01
    )
    grad_y = [func(x) for x in gradList]

    plt.plot(x, y, label="f(x)")

    # Plot gradient descent path
    plt.plot(
        gradList,
        grad_y,
        'ro-',
        label="Gradient Descent"
    )

    # Mark starting point
    plt.scatter(
        gradList[0],
        grad_y[0],
        s=100,
        label="Start"
    )

    # Mark final point
    plt.scatter(
        best_x,
        func(best_x),
        s=100,
        label="Minimum"
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Gradient Descent")
    plt.legend()
    plt.grid()

    plt.show()




if __name__ == '__main__':
    func1()