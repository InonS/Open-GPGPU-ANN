"""
http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/
"""
from logging import DEBUG, basicConfig, getLogger, info
from sys import stdout

from keras.activations import softmax, tanh
from keras.layers import Activation, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.utils import plot_model
from numpy.random import random
from pydot import Dot

basicConfig(level=DEBUG, stream=stdout)


def build_model(data_dim, nb_classes, n_fully_connected_nodes=1 << 6,
                hidden_activation=Activation(tanh), out_activation=Activation(softmax), dropout=0.5):
    droput_layer = Dropout(dropout)

    model = Sequential()

    # Dense(n_fully_connected_nodes) is a fully-connected layer with n_fully_connected_nodes hidden units.
    # in the first layer, you must specify the expected input data shape: here, (data_dim)-dimensional vectors.
    model.add(Dense(n_fully_connected_nodes, input_dim=data_dim, kernel_initializer='uniform'))
    model.add(hidden_activation)
    model.add(droput_layer)
    model.add(Dense(n_fully_connected_nodes, kernel_initializer='uniform'))
    model.add(hidden_activation)
    model.add(droput_layer)
    model.add(Dense(nb_classes, kernel_initializer='uniform'))
    model.add(out_activation)

    model.compile(optimizer='sgd', loss=categorical_crossentropy, metrics=[categorical_accuracy])

    return model


def run(data_dim=1 << 4, nb_classes=1 << 2, n_train_samples=1 << 10, n_test_samples=1 << 7, batch_size=1 << 4,
        epochs=1 << 5):
    model = build_model(data_dim, nb_classes)

    # generate dummy training data
    x_train = random((n_train_samples, data_dim))
    y_train = random((n_train_samples, nb_classes))

    # generate dummy test data
    x_test = random((n_test_samples, data_dim))
    y_test = random((n_test_samples, nb_classes))

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    maybe_plot_model(model)

    metrics = zip(model.metrics_names, model.evaluate(x_test, y_test, batch_size=batch_size))
    return {pair[0]: pair[1] for pair in metrics}


def maybe_plot_model(model):
    try:
        Dot.create(Dot())
        model_plot_filename = '.'.join((__name__, "png"))
        plot_model(model, to_file=model_plot_filename, show_shapes=True)
    except RuntimeError:
        pass


def main():
    metrics = run()
    for h in getLogger().handlers:
        h.flush()
    info("metrics: {}".format(metrics))


if __name__ == '__main__':
    main()
