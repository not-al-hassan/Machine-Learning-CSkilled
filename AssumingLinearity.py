import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.methods.selectn import DataFrame
from sklearn.preprocessing import MinMaxScaler

"""""
- All the time we are assuming that the data comes from a line so 
we will use LR .. but after visualization we notice that only the first
feature is in linear relationship with target

- Lets now retrain the model with these feature and see results  !


- When using the 3 features :

 Minimum score is 0.0031116303614825824 and it found when using max_iter : 300 and learning rate 0.1 and precision 0.01

- When using only the first feature :

Minimum score is 0.004045635517629554 and it found when using max_iter : 300 and learning rate 0.1 and precision 0.01


Although Feature 1 shows the clearest linear relationship with the target,
the model trained using all three features achieves a lower MSE (0.00311 vs. 0.00405).

This indicates that Features 2 and/or 3 provide additional information that helps reduce 
the prediction error, even though their individual plots may not appear strongly linear.

"""""


def gradient_descent(x, y, states, f_der, max_iter = 1000, learning_rate = 0.1, precision = 0.001):

    cur = np.array(states)
    last = cur + 100 + precision

    costs = []

    for _ in range(max_iter):
        last = cur.copy()
        cost = MSE(x, y, cur)
        costs.append(cost)
        gradient = f_der(x, y, cur)
        cur = (cur - gradient * learning_rate)

        if np.all(np.abs(cur - last)) < precision:
            break


    return cur, costs



def MSE(x, y, weights):
    prediction = x @ weights
    error = prediction - y
    return error.T @ error / (2 * len (y))

def MSE_der(x, y, weights):
    prediction = x @ weights
    error = prediction - y
    return x.T @ error / len(y)

def trial():

    df = pd.read_csv('dataset_200x4_regression.csv')

    y = df['Target']

    x = df.drop(columns='Target')
    x = x.iloc[:, [0]]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x = np.column_stack((np.ones(len(x)), x))

    states = np.array([1.0, 1.0])



    max_iters = [100, 200, 300]
    learning_rates = [0.1, 0.01]
    precisions = [0.01, 0.0001]



    best_max_iter, best_learning_rate, best_precision = -1, -1, -1
    score = float('inf')

    for max_iter in max_iters:
        for learning_rate in learning_rates:
            for precision in precisions:
                weights, costs = gradient_descent(x, y, states, MSE_der, max_iter, learning_rate, precision)
                cost = MSE(x, y, weights)

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

    print(
        f"Minimum score is {score} and it found when using max_iter : {best_max_iter} and learning rate {best_learning_rate} and precision {best_precision}")


if __name__ == '__main__':
    print(0.004045635517629554 < 0.0031116303614825824)
    trial()