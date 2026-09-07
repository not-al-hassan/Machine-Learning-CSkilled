
# Homework 1 --> Implementation of Linear Regression using Gradient Descent.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def gradient_descent(x, y, states, f_der,
                     max_iter=3,
                     learning_rate=0.1,
                     precision=0.00001):

    cur = np.array(states, dtype=float)
    states_list = [cur.copy()]

    costs = []


    for _ in range(max_iter):

        gradient = f_der(x, y, cur)

        cost = MSE(x, y, cur)
        costs.append(cost)

        cur -= learning_rate * gradient

        states_list.append(cur.copy())

    return cur, costs


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

    df = pd.read_csv('dataset_200x4_regression.csv')

    n, features = df.shape

    y = df['Target']
    x = df.drop(columns='Target')

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)


    X = np.column_stack((np.ones(len(x)), x))


    states = np.array([1.0,1.0,1.0,1.0])


    max_iters = [100, 200, 300]
    learning_rates = [0.1, 0.01]
    precisions = [0.01, 0.0001]

    best_max_iter , best_learning_rate, best_precision = -1, -1, -1
    score = float('inf')



    for max_iter in max_iters:
        for learning_rate in learning_rates:
            for precision in precisions:
                weights, costs = gradient_descent(X, y, states, MSE_der,  max_iter, learning_rate, precision)
                cost = MSE (X, y,weights)

                iterations = np.arange(max_iter)

                # plt.plot(iterations, costs)
                # plt.xlabel('The number of iterations')
                # plt.ylabel('costs')
                # plt.show()

                if cost < score:
                    score = cost
                    best_max_iter = max_iter
                    best_learning_rate = learning_rate
                    best_precision = precision


    print(f"Minimum score is {score} and it found when using max_iter : {best_max_iter} and learning rate {best_learning_rate} and precision {best_precision}")
