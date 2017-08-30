# TODO tensorflow.contrib.stat_summarizer
# TODO tensorflow.contrib.bayesflow

from logging import DEBUG, basicConfig, debug, info
from math import log2
from sys import getsizeof, stdout
from tempfile import mkdtemp

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.nn_ops import bias_add, xw_plus_b
from tqdm import trange

basicConfig(level=DEBUG, stream=stdout)


def read_data():
    """
    1. get MNIST data
        a. 28x28 b/w pixel pictures of handwritten digits
        b. 50,000 training samples and 5,000 test samples
        c. Multi-class (10 classes: '0', '1', ..., '9') classification
        d. one-hot encoding: '0' = [1, 0, 0, ..., 0] ; '1' = [0, 1, 0, ..., 0] ; ... ; '9' = [0, 0, 0, ..., 1]
    """
    tempd = mkdtemp()
    debug("tempd = %s" % tempd)
    mnist_ = input_data.read_data_sets(tempd, one_hot=True)
    num_train_samples_ = mnist_.train.num_examples
    return mnist_, num_train_samples_


mnist, NUM_TRAIN_SAMPLES = read_data()
debug("mnist = %s" % str(mnist))
info("NUM_TRAIN_SAMPLES = %d" % NUM_TRAIN_SAMPLES)


def data_shape():
    """
    2. Build model:
        a. Input layer: 28*28 = 784 float-valued nodes (flattened 2-D picture into a 1-D array)
        b. Output layer: 10 nodes
        c. e.g. 3 hidden layers: e.g. 500 nodes each
    """
    image_dims = [28, 28]
    n_pixel_rows, n_pixel_columns = image_dims

    n_channels = 1  # color / depth (one if simple, monochromatic image)

    n_label_classes = len(mnist.train.labels[0])

    return n_pixel_columns, n_pixel_rows, n_channels, n_label_classes


CHUNK_SIZE, N_CHUNKS, N_CHANNELS, N_LABEL_CLASSES = data_shape()


def preprocess(data):
    x_ = data
    x_ = tf.transpose(x_, [1, 0, 2])  # acylic permutation of dimensions
    x_ = tf.reshape(x_, [-1, CHUNK_SIZE])  # flatten
    x_ = tf.split(x_, N_CHUNKS)
    return x_


def model_params():
    """
    0. Starting with 28*28 pixels input.
    1. Each subsampling operation reduces the feature-map dimensionality.
    For example, a single subsampling operation with size 2 will make the input of the next layer half as big
    2. A second such operation will make the input of the following (e.g. fully-connected) layer a quarter of original
    That is, (28 / 2 / 2) * (28 / 2 / 2) = 7 * 7
    """
    # Convolution layers
    window_width_conv = 5  # number of pixels in window width of conolution layers
    window_height_conv = 5  # number of pixels in window hight of conolution layers
    fm_conv = [1 << 5, 1 << 6]  # feature map size for Convolution layers

    # Fully-Connected layer
    window_width_fc = 7  # number of pixels in window width of fully-connected layer
    window_height_fc = 7  # number of pixels in window height of fully-connected layer
    fm_fc = window_width_fc * window_height_fc * fm_conv[1]  # feature map size for fully-connected layer
    n_nodes_fc = 1 << 10  # number of nodes in fully-connected layer

    weights = {
        'W_conv0': tf.Variable(tf.random_normal([window_width_conv, window_height_conv, N_CHANNELS, fm_conv[0]])),
        'W_conv1': tf.Variable(
            tf.random_normal([window_width_conv, window_height_conv, fm_conv[0], fm_conv[1]])),
        'W_fc': tf.Variable(tf.random_normal([fm_fc, n_nodes_fc])),
        'W_out': tf.Variable(tf.random_normal([n_nodes_fc, N_LABEL_CLASSES]))}

    biases = {'b_conv0': tf.Variable(tf.random_normal([fm_conv[0]])),
              'b_conv1': tf.Variable(tf.random_normal([fm_conv[1]])),
              'b_fc': tf.Variable(tf.random_normal([n_nodes_fc])),
              'b_out': tf.Variable(tf.random_normal([N_LABEL_CLASSES]))}

    return weights, biases


def conv2d(data, w, samples=1, rows=1, columns=1, outputs=1):
    """
    Extract high-order features
    """
    # To include depth see conv3d
    return tf.nn.conv2d(data, w,
                        strides=[samples, rows, columns, outputs],
                        padding="SAME")


def maxpool2d(data,
              window_samples=1, window_rows=2, window_columns=2, window_outputs=1,
              stride_samples=1, stride_rows=2, stride_columns=2, stride_outputs=1):
    """
    Subsample
    """
    return tf.nn.max_pool(data,
                          ksize=[window_samples, window_rows, window_columns, window_outputs],
                          strides=[stride_samples, stride_rows, stride_columns, stride_outputs],
                          padding="SAME")


def conv_op(prev_layer, weight, bias):
    convolved = conv2d(prev_layer, weight)
    biased = bias_add(convolved, bias)
    activated = tf.nn.relu(biased)
    pooled = maxpool2d(activated)
    return pooled


def fc_op(prev_layer, weight, bias):
    n_fc_inputs, n_fc_outputs = weight.get_shape().as_list()
    fc_input = tf.reshape(prev_layer, shape=[-1, n_fc_inputs])
    fc_preactivation = xw_plus_b(fc_input, weight, bias)
    fc = tf.nn.relu(fc_preactivation)
    return fc


def model_ops(data, weights, biases, keep_rate=0.8):
    # reshape data to have shape of: (num_samples) * N_PIXEL_ROWS * N_PIXEL_COLUMNS * N_CHANNELS
    # (num_samples is inferred)
    x_ = tf.reshape(data, shape=[-1, N_CHUNKS, CHUNK_SIZE, N_CHANNELS])
    conv0 = conv_op(x_, weights['W_conv0'], biases['b_conv0'])
    conv1 = conv_op(conv0, weights['W_conv1'], biases['b_conv1'])
    fc = fc_op(conv1, weights['W_fc'], biases['b_fc'])
    keep_prob = tf.constant(keep_rate)
    dropout = tf.nn.dropout(fc, keep_prob)
    out = xw_plus_b(dropout, weights['W_out'], biases['b_out'])
    return out


def model(data):
    """
    Deep MNIST for experts (TensorFlow)
    """
    params = model_params()
    return model_ops(data, *params)


x = tf.placeholder(tf.float32, shape=[None, N_CHUNKS, CHUNK_SIZE], name='input')
y = tf.placeholder(tf.float32, name='output')


def get_batch_size(num_train_samples_, min_num_batches=10):
    kibi = 1 << 10
    gibi = kibi * kibi * kibi
    ram_size = 4 * gibi

    sample_size = getsizeof(mnist.train.images[0])
    batch_size = int(ram_size / sample_size)
    if num_train_samples_ <= batch_size:
        batch_size = int(num_train_samples_ / min_num_batches)
    batch_size = max(1 << int(log2(batch_size)), 1)  # make sure BATCH_SIZE is an integer power of 2
    return batch_size


def train(cost, optimizer, sess: tf.Session, train_dataset, max_epochs=10):
    """
    2.5 min / epoch, single-digit percent error (in accuracy) after ~10 epochs
    """
    fetches = [optimizer, cost]

    # Out-of-core training: Batches
    batch_size = get_batch_size(NUM_TRAIN_SAMPLES)
    info("batch_size = %d" % batch_size)
    total_batches = int(NUM_TRAIN_SAMPLES / batch_size)

    epoch_postfix = {"epoch_loss": 0}
    epochs_iter = trange(max_epochs, desc="epochs", file=stdout, mininterval=2, unit="epoch")
    for epoch in epochs_iter:

        epoch_loss = 0
        batch_postfix = {"batch_cost": 0}
        batches_iter = trange(total_batches, desc="epoch %d / %d batches" % (epoch, max_epochs), file=stdout,
                              mininterval=2,
                              unit="batch", postfix=batch_postfix)
        for _ in batches_iter:
            batch_x, batch_y = train_dataset.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, N_CHUNKS, CHUNK_SIZE))

            _, batch_cost = sess.run(fetches, feed_dict={x: batch_x, y: batch_y})
            batch_postfix["batch_cost"] = batch_cost
            batches_iter.set_postfix(batch_postfix)
            epoch_loss += batch_cost

        epoch_postfix["epoch_loss"] = epoch_loss
        epochs_iter.set_postfix(epoch_postfix)


def test(prediction, test_dataset):
    correct_ones = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_ones, tf.float32))

    test_data = {x: test_dataset.images.reshape((-1, N_CHUNKS, CHUNK_SIZE)),
                 y: test_dataset.labels}  # note placeholder keys
    info("Accuracy: %g" % accuracy.eval(test_data))


def run(data):
    # data = preprocess(data)
    prediction = model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    train_dataset = mnist.train
    test_dataset = mnist.test

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(cost, optimizer, sess, train_dataset, max_epochs=1)
        test(prediction, test_dataset)


def main():
    run(x)


if __name__ == '__main__':
    main()
