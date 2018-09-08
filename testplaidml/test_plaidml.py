#!/usr/bin/env python
from logging import DEBUG, basicConfig, info
from os import putenv
from sys import stdout
from time import time

from numpy import repeat, uint8
from plaidml.keras import install_backend
from tqdm import trange

# Install the plaidml backend
install_backend()

from keras.applications import VGG19
from keras.datasets import cifar10


def data(batch_size):
    (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
    x_train = x_train[:batch_size]
    x_train = repeat(repeat(x_train, 7, axis=1), 7, axis=2)
    return x_train, x_test, y_test_cats


def compile_model():
    model = VGG19()
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def infer(model, x_train, batch_size, n_batches: uint8 = 10):
    time_inference_init(batch_size, model, x_train)
    time_inference(batch_size, model, n_batches, x_train)


def time_inference_init(batch_size, model, x_train):
    info("Running initial batch (compiling tile program):")
    start = time()
    model.predict(x=x_train, batch_size=batch_size)
    info("Tile program compilation took {} seconds".format(time() - start))


def time_inference(batch_size, model, n_batches, x_train):
    info("Now start the clock and run %d batches:" % n_batches)
    start = time()
    for _ in trange(n_batches):
        model.predict(x=x_train, batch_size=batch_size)
    info("Inference took {} seconds".format(time() - start))


def set_env(gpu_alloc=0.5, gpu_max_heap=0.5):
    """
    https://github.com/plaidml/plaidml/issues/67#issuecomment-350520463
    """
    assert gpu_alloc <= 1
    assert gpu_max_heap <= 1
    # GPU_FORCE_64BIT_PTR=1
    # GPU_USE_SYNC_OBJECTS=1
    putenv("GPU_MAX_ALLOC_PERCENT", str(uint8(100 * gpu_alloc)))
    putenv("GPU_SINGLE_ALLOC_PERCENT", str(uint8(100 * gpu_alloc)))
    putenv("GPU_MAX_HEAP_SIZE", str(uint8(100 * gpu_max_heap)))


def main(batch_size=1 << 3):
    set_env()
    x_train, x_test, y_test_cats = data(batch_size)
    model = compile_model()
    infer(model, x_train, batch_size)


if __name__ == '__main__':
    basicConfig(stream=stdout, level=DEBUG)
    main()
