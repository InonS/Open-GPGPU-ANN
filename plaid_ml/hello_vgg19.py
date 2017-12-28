#!/usr/bin/env python

from time import time

from numpy import repeat
from plaidml.keras import install_backend

# Install the plaidml backend
install_backend()

from keras.applications import VGG19
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
x_train = repeat(repeat(x_train, 7, axis=1), 7, axis=2)
model = VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time()
for i in range(10):
    _ = model.predict(x=x_train, batch_size=batch_size)
print("Ran in {} seconds".format(time() - start))
