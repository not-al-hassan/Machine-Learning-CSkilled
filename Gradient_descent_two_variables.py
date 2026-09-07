import numpy as np
import matplotlib.pyplot as plt


def gradient_descent( func_der_x, func_der_y, start_x, start_y, max_iter = 5000, learning_rate = 0.01, precision = 0.001):
    cur_xy = np.array([start_x, start_y])
    last_xy = np.array([float('inf'), float('inf')])
    lst = [cur_xy]


    for _ in range(max_iter):
        last_xy = cur_xy.copy()

        f_x = func_der_x(cur_xy[0], cur_xy[1])
        f_y = func_der_y(cur_xy[0], cur_xy[1])

        gradient = np.array([f_x, f_y])

        cur_xy -= (gradient * learning_rate)

        if np.all(np.abs(cur_xy - last_xy) < precision):
            break

        lst.append(cur_xy)

    return lst, cur_xy



def trial():

    def func(x, y):
        return 3 * ((x + 2) ** 2) + ((y - 1) ** 2)

    def func_der_x(x,y):
        return 6 * (x + 2)

    def func_der_y(x,y):
        return 2 * (y - 1)

    inital_x, inital_y = -5.0, 2.5

    xy_list , cur_xy = gradient_descent(func_der_x, func_der_y, inital_x, inital_y)

    print(f"The Minimum of the function exists at x = {cur_xy[0]} and at y = {cur_xy[1]}")



if __name__ == '__main__':
    trial()