from logging import DEBUG, basicConfig, debug
from sys import stdout

from keras.activations import relu, softmax
from keras.backend import categorical_crossentropy, expand_dims, one_hot
from keras.datasets.mnist import load_data
from keras.engine import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop

SAME = 'SAME'


def image_input(img_width=1 << 8, img_height=1 << 8, n_channels=3):
    """
    :param img_width:
    :param img_height:
    :param n_channels:  e.g. Monochrome = 1, RGB = 3, Hyperspectral > 1, etc.
    :return:
    """
    return Input(shape=(img_width, img_height, n_channels), name="mnist_input")


def conv2d_of(window_length, n_conv_filters=int(1 << 6)):
    return Conv2D(n_conv_filters, (window_length, window_length), padding=SAME, activation=relu)


def conv_tower(window_length, input_img):
    first_layer = conv2d_of(1)(input_img)
    return conv2d_of(window_length)(first_layer)


def preprocess(train_dataset, test_dataset, n_classes_expected):
    x_train, y_train = train_dataset
    train_features_shape = x_train.shape
    train_labels_shape = y_train.shape
    debug("train features shape = {}, train labels shape = {}".format(train_features_shape, train_labels_shape))

    # Image Augmentation
    # datagen = ImageDataGenerator()
    # datagen.fit(x_train)

    x_test, y_test = test_dataset

    y_train = one_hot(y_train, n_classes_expected)
    y_test = one_hot(y_test, n_classes_expected)

    # add n_channels = 1 if missing
    if len(train_features_shape) < 4:
        x_train = expand_dims(x_train)
        x_test = expand_dims(x_test)

    # https://github.com/fchollet/keras/issues/7756
    n_train_samples, img_width, img_heigth, n_channels = x_train.shape.as_list()

    image_dims = img_width, img_heigth, n_channels
    train_dataset = x_train, y_train
    test_dataset = x_test, y_test
    return image_dims, train_dataset, test_dataset


def run():
    train_dataset, test_dataset = load_data()

    n_classes_expected = 10
    image_dims, train_dataset, test_dataset = preprocess(train_dataset, test_dataset, n_classes_expected)

    img_width, img_heigth, n_channels = image_dims
    x_train, y_train = train_dataset
    x_test, y_test = test_dataset

    input_img = image_input(img_width, img_heigth, n_channels)
    cnn = conv_tower(3, input_img)
    flattened = Flatten()(cnn)
    predictions = Dense(n_classes_expected, activation=softmax)(flattened)

    model = Model(input_img, predictions, name="CNN")
    model.compile(RMSprop(), categorical_crossentropy, metrics=[categorical_accuracy])

    model.fit(x=x_train, y=y_train, batch_size=None, epochs=1, validation_split=0.3)
    model.evaluate(x_test, y_test, batch_size=None)


def main():
    basicConfig(level=DEBUG, stream=stdout)
    run()


if __name__ == '__main__':
    main()
