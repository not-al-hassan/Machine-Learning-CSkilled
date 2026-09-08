import numpy as np

EPS = 1e-4


def MSE(x, y, weights):

    prediction = x @ weights
    error = prediction - y

    return (error.T @ error) / (len(y))



def Analatical_Gradient_MSE(x, y, weights):

    prediction = x @ weights
    error = prediction - y

    return (2 * x.T @ error) / (len (y))


def Gradient_Check(x, y, weights):

    numerical_gradient = np.zeros_like(weights)

    for i in range(len(weights)):

        weights_plus = weights.copy()
        weights_minus = weights.copy()

        weights_plus[i] += EPS
        weights_minus[i] -= EPS

        numerical_gradient[i] = (
            MSE(x, y, weights_plus)
            -
            MSE(x, y, weights_minus)
        ) / (2 * EPS)

    analytical_gradient = Analatical_Gradient_MSE(
        x, y, weights
    )

    return numerical_gradient, analytical_gradient


def is_my_gradient_right(numerical_gradient, analatical_gradinet):
    return np.all((np.abs(numerical_gradient - analatical_gradinet)) <= EPS)



if __name__ == '__main__':

    x = np.array([1,2,3,4,5])
    y = np.array([1,2,3,4, 5])

    x = np.column_stack((np.ones(len(x)), x))

    states = np.ones(2)

    num, ana = Gradient_Check(x, y, states)


    if is_my_gradient_right(num, ana):
        print("Your Gradient is Right !")
    else:
        print("Wrong Gradient Check it Again !")




