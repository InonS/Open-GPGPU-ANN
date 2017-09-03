from functools import partial
from logging import DEBUG, basicConfig
from sys import stdout

from keras.activations import relu, softmax
from keras.backend import categorical_crossentropy
from keras.datasets.mnist import load_data
from keras.engine import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, concatenate
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop

from testkeras.testKerasFunctionalMinimal import preprocess

SAME = 'SAME'


def image_input(img_height=1 << 8, img_width=1 << 8, n_channels=3):
    """
    :param img_width:
    :param img_height:
    :param n_channels:  e.g. Monochrome = 1, RGB = 3, Hyperspectral > 1, etc.
    :return:
    """
    return Input(shape=(img_height, img_width, n_channels))


def conv2d_of(window_length, n_conv_filters=int(1 << 6)):
    return Conv2D(n_conv_filters, (window_length, window_length), padding=SAME, activation=relu)


def conv_tower(window_length, input_img):
    first_layer = conv2d_of(1)(input_img)
    return conv2d_of(window_length)(first_layer)


def maxpool2d_of(pool_length):
    return MaxPooling2D((pool_length, pool_length), strides=(1, 1), padding=SAME)


def pool_tower(pool_length, input_img):
    pool_layer = maxpool2d_of(pool_length)(input_img)
    return conv2d_of(1)(pool_layer)


def inception_module(input_img, small_window=3, large_window=5, pool_length=3):
    tower_1 = partial(conv_tower, small_window)  # small window tower
    tower_2 = partial(conv_tower, large_window)  # large window tower
    tower_3 = partial(pool_tower, pool_length)  # subsampled tower
    return concatenate([tower_1(input_img), tower_2(input_img), tower_3(input_img)], axis=1)


def build_model(image_dims, n_classes_expected):
    input_img = image_input(*image_dims)
    inception = inception_module(input_img)
    flatten = Flatten()(inception)
    predictions = Dense(n_classes_expected, activation=softmax)(flatten)

    model = Model(inputs=input_img, outputs=predictions, name="inception")
    model.compile(RMSprop(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    return model


def run():
    n_classes_expected = 10
    image_dims, train_dataset, test_dataset = preprocess(*load_data(), n_classes_expected)
    model = build_model(image_dims, n_classes_expected)
    model.fit(*train_dataset, batch_size=None, epochs=1, validation_split=0.3)
    model.evaluate(*test_dataset, batch_size=None)


def main():
    basicConfig(level=DEBUG, stream=stdout)
    run()


if __name__ == '__main__':
    main()
