from logging import DEBUG, basicConfig, debug, info
from os.path import join as path_join
from sys import stdout

from keras.datasets.mnist import load_data
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import pi, rad2deg

from testkeras.testKeras import metrics_dict
from testkeras.testKerasFunctional import DATA_DIR, build_model, persistence_filename
from testkeras.testKerasFunctionalMinimal import preprocess

MODEL_FILEPATH = path_join(DATA_DIR, persistence_filename(str(__file__)))
WEIGHTS_FILEPATH = path_join(DATA_DIR, persistence_filename(str(__file__), sub="weights"))


def augmented_flow(train_dataset, batch_size, fill_mode='constant'):
    # Image Augmentation:
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    eighth_circle = pi / 4  # in radians
    datagen = ImageDataGenerator(rotation_range=rad2deg(eighth_circle),
                                 width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=eighth_circle, zoom_range=0.2,
                                 fill_mode=fill_mode,
                                 horizontal_flip=True, vertical_flip=True)
    x_train, y_train = train_dataset
    datagen.fit(x_train)
    return datagen.flow(x_train, y=y_train, batch_size=batch_size)


def get_model(io_dims, train_dataset, batch_size=1 << 5, epochs=1, n_flow_steps=None):
    try:
        debug("Attempting to load model from %s" % MODEL_FILEPATH)
        model = load_model(MODEL_FILEPATH)
    except (OSError, ValueError):
        model = build_model(*io_dims)
        try:
            debug("Attempting to load model weights from %s" % WEIGHTS_FILEPATH)
            model.load_weights(WEIGHTS_FILEPATH)
        except OSError:
            if n_flow_steps:
                steps_per_epoch = n_flow_steps / batch_size
                model.fit_generator(train_dataset, steps_per_epoch,
                                    epochs=epochs)  # , workers=cpu_count() - 2, use_multiprocessing=True)
            else:
                model.fit(*train_dataset, batch_size=batch_size, epochs=epochs, validation_split=0.3)
            debug("Saving model weights to %s" % WEIGHTS_FILEPATH)
            model.save_weights(WEIGHTS_FILEPATH)
        debug("Saving model to %s" % MODEL_FILEPATH)
        model.save(MODEL_FILEPATH)
    return model


def run():
    n_classes_expected = 10
    batch_size = 1 << 5
    image_dims, train_dataset, test_dataset = preprocess(*load_data(), n_classes_expected)
    n_train_labels = len(train_dataset[1])
    model = get_model((image_dims, n_classes_expected), augmented_flow(train_dataset, batch_size),
                      batch_size=batch_size, epochs=1, n_flow_steps=n_train_labels)
    metric_values = model.evaluate(*test_dataset, batch_size=batch_size)
    results = metrics_dict(model, metric_values)
    info(results)
    return results


def main():
    basicConfig(level=DEBUG, stream=stdout)
    run()


if __name__ == '__main__':
    main()
