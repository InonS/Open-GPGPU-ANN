"""
1. Build feed-forward net: input -> weight -> hidden layer 1 (activation function) -> weight -> hidden l2 (activation) -> weights -> output
2. Compare output to label with cost (or loss) function (e.g. cross entropy)
3. Minimize cost using optimizer (e.g. ADAM, SGD, AdaGrad, etc.)
4. Propagate the correction back to the weights in the original network: Back-propagation
1 + 2 + 3 + 4 = epoch
repeat until convergence (if not monotonous, use early-stopping)
"""
from logging import basicConfig, debug, info, INFO
from math import log2
from sys import stdout, getsizeof
from tempfile import mkdtemp

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import static_rnn
from tqdm import trange

basicConfig(level=INFO, stream=stdout)

"""
 1. get MNIST data
    a. 28x28 b/w pixel pictures of handwritten digits 
    b. 50,000 training samples and 5,000 test samples
    c. Multi-class (10 classes: '0', '1', ..., '9') classification
    d. one-hot encoding: '0' = [1, 0, 0, ..., 0] ; '1' = [0, 1, 0, ..., 0] ; ... ; '9' = [0, 0, 0, ..., 1]
"""
tempd = mkdtemp()
debug("tempd = %s" % tempd)
mnist = input_data.read_data_sets(tempd, one_hot=True)
debug("mnist = %s" % str(mnist))
NUM_TRAIN_SAMPLES = mnist.train.num_examples
info("NUM_TRAIN_SAMPLES = %d" % NUM_TRAIN_SAMPLES)

# Out-of-core training: Batches
SAMPLE_SIZE = getsizeof(mnist.train.images[0])
KIBI = 1 << 10
GIBI = KIBI * KIBI * KIBI
RAM_SIZE = 4 * GIBI
BATCH_SIZE = int(RAM_SIZE / SAMPLE_SIZE)
if NUM_TRAIN_SAMPLES <= BATCH_SIZE:
    MIN_NUM_BATCHES = 10
    BATCH_SIZE = int(NUM_TRAIN_SAMPLES / MIN_NUM_BATCHES)
BATCH_SIZE = max(1 << int(log2(BATCH_SIZE)), 1)  # make sure BATCH_SIZE is an integer power of 2
info("BATCH_SIZE = %d" % BATCH_SIZE)
"""
2. Build model:
    a. Input layer: 28*28 = 784 float-valued nodes (flattened 2-D picture into a 1-D array)
    b. Output layer: 10 nodes
    c. e.g. 3 hidden layers: e.g. 500 nodes each
"""
IMAGE_DIMS = [28, 28]
N_PIXEL_ROWS, N_PIXEL_COLUMNS = IMAGE_DIMS
CHUNK_SIZE = N_PIXEL_COLUMNS
N_CHUNKS = N_PIXEL_ROWS

N_LABEL_CLASSES = len(mnist.train.labels[0])

N_HIDDEN_LAYERS = 0
n_hidden_layer_nodes = [500] * N_HIDDEN_LAYERS

N_UNITS_IN_LSTM_CELL = 1 << 7

# placeholders for explicit values we know we'll be handling: Input & output
# specify shapes for the benefit of runtime validation
x = tf.placeholder(tf.float32, shape=[None, N_CHUNKS, CHUNK_SIZE], name='input')
y = tf.placeholder(tf.float32, name='output')


def model(data):
    x_ = preprocess(data)

    # RNN model with LSTM cell with N_UNITS_IN_LSTM_CELL
    lstm_cell = rnn_cell.BasicLSTMCell(N_UNITS_IN_LSTM_CELL, state_is_tuple=True)
    outputs, states = static_rnn(lstm_cell, x_, dtype=tf.float32)

    # connections to output layer (N_UNITS_IN_LSTM_CELL -> N_LABEL_CLASSES)
    layer = {'weights': tf.Variable(tf.random_normal([N_UNITS_IN_LSTM_CELL, N_LABEL_CLASSES])),
             'biases': tf.Variable(tf.random_normal([N_LABEL_CLASSES]))}
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def preprocess(data):
    x_ = data
    x_ = tf.transpose(x_, [1, 0, 2])  # acylic permutation of dimensions
    x_ = tf.reshape(x_, [-1, CHUNK_SIZE])  # flatten
    x_ = tf.split(x_, N_CHUNKS)
    return x_


def train(cost, optimizer, sess: tf.Session, train_dataset, max_epochs=3):
    """

    :param cost:
    :param optimizer:
    :param sess:
    :param train_dataset:
    :param max_epochs: [one-shot -> (20 seconds, 43% accuracy), 3 epochs -> (1 minute, 71% accuracy), 10 epochs -> (4 minutes, >93% accuracy)]
    :return:
    """
    fetches = [optimizer, cost]
    total_batches = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)

    epoch_postfix = {"epoch_loss": 0}
    epochs_iter = trange(max_epochs, desc="epochs", file=stdout, mininterval=2, unit="epoch")
    for epoch in epochs_iter:

        epoch_loss = 0
        batch_postfix = {"batch_cost": 0}
        batches_iter = trange(total_batches, desc="epoch %d / %d batches" % (epoch, max_epochs), file=stdout,
                              mininterval=2,
                              unit="batch", postfix=batch_postfix)
        for _ in batches_iter:
            batch_x, batch_y = train_dataset.next_batch(BATCH_SIZE)
            batch_x = batch_x.reshape((BATCH_SIZE, N_CHUNKS, CHUNK_SIZE))

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


def run(x_):
    prediction = model(x_)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    train_dataset = mnist.train
    test_dataset = mnist.test

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(cost, optimizer, sess, train_dataset, 1)
        test(prediction, test_dataset)


run(x)
