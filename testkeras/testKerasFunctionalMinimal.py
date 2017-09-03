from logging import DEBUG, basicConfig, debug
from sys import stdout

from keras.activations import relu, softmax
from keras.backend import categorical_crossentropy
from keras.datasets.mnist import load_data
from keras.engine import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from numpy import expand_dims

SAME = 'SAME'


def preprocess(train_dataset, test_dataset, n_classes_expected):
    x_train, y_train = train_dataset
    train_features_shape = x_train.shape
    train_labels_shape = y_train.shape
    debug("train features shape = {}, train labels shape = {}".format(train_features_shape, train_labels_shape))

    # Image Augmentation:
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # datagen = ImageDataGenerator()
    # datagen.fit(x_train)

    x_test, y_test = test_dataset

    y_train = to_categorical(y_train, n_classes_expected)
    y_test = to_categorical(y_test, n_classes_expected)

    # add n_channels = 1 if missing
    if len(train_features_shape) < 4:
        x_train = expand_dims(x_train, -1)
        x_test = expand_dims(x_test, -1)

    # https://github.com/fchollet/keras/issues/7756
    n_train_samples, img_width, img_heigth, n_channels = x_train.shape

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

    input_img = Input(shape=(img_width, img_heigth, n_channels))
    model = build_model(input_img, n_classes_expected)

    model.fit(x=x_train, y=y_train, batch_size=None, epochs=1, validation_split=0.3)
    model.evaluate(x_test, y_test, batch_size=None)


def build_model(input_img, n_classes_expected, n_conv_filters=1 << 6, window_length=3):
    cnn = Conv2D(n_conv_filters, (window_length, window_length), padding=SAME, activation=relu)(input_img)
    flattened = Flatten()(cnn)
    predictions = Dense(n_classes_expected, activation=softmax)(flattened)

    model = Model(input_img, predictions, name="CNN")
    model.compile(RMSprop(), categorical_crossentropy, metrics=[categorical_accuracy])
    return model


def main():
    basicConfig(level=DEBUG, stream=stdout)
    run()


if __name__ == '__main__':
    main()
