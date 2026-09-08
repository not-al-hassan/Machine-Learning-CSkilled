
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':

    mnist = fetch_openml('mnist_784', version=1, as_frame=True)

    x = mnist.data.astype('float') / 255.0

    x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)

    autoEncoder = MLPRegressor(max_iter=5000, random_state=42,
                               hidden_layer_sizes=(32,), activation='relu', solver='adam')

    autoEncoder.fit(x_train,x_train)
    prediction = autoEncoder.predict(x_test)

    mse = mean_squared_error(x_test, prediction)

    print(mse)



