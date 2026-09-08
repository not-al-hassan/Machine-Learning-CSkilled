import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma.extras import column_stack


def Normal_equation(x, y):
    theta = np.linalg.solve(x.T @ x ,   x.T @ y)
    return theta


if __name__ == '__main__':

    df = pd.read_csv('dataset_200x4_regression.csv')

    y = df['Target']
    x = df.drop(columns = 'Target')

    X = np.column_stack((np.ones(len(x)),x)) # 4 columns c0, w1, w2, w3


    weights = Normal_equation(X, y)

    c = weights[0]
    m1 = weights[1]
    m2 = weights[2]
    m3 = weights[3]

    prediction = X @ weights

    feat1 = x.iloc[:, 0]

    print(prediction.shape)

    plt.scatter(x.iloc[:, 0], y, label = 'Actual')
    plt.scatter(x.iloc[:, 0], prediction, label = 'Predicted')

    plt.legend()
    plt.show()


    plt.scatter(x.iloc[:, 1], y, label = 'Actual')
    plt.scatter(x.iloc[:, 1], prediction, label = 'Predicted')

    plt.legend()
    plt.show()

    plt.scatter(x.iloc[:, 2], y, label = 'Actual')
    plt.scatter(x.iloc[:, 2], prediction, label = 'Predicted')

    plt.legend()
    plt.show()