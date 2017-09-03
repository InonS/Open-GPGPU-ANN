from logging import DEBUG, basicConfig, info
from sys import getsizeof, stdout

import tensorflow as tf
from numpy import array

from sentdex_nn_ml_tutorial.create_sentiment_featuresets import unpickle_design_matrix

basicConfig(level=DEBUG, stream=stdout)
x_train, y_train, x_test, y_test = unpickle_design_matrix()

NUM_TRAIN_SAMPLES = len(x_train)
info(NUM_TRAIN_SAMPLES)

# Out-of-core training: Batches
SAMPLE_SIZE = getsizeof(y_train[0])
KIBI = 1 << 10
GIBI = KIBI * KIBI * KIBI
RAM_SIZE = 4 * GIBI
BATCH_SIZE = int(RAM_SIZE / SAMPLE_SIZE)
if NUM_TRAIN_SAMPLES <= BATCH_SIZE:
    BATCH_SIZE = int(NUM_TRAIN_SAMPLES / 100)
info(BATCH_SIZE)

N_INPUT_LAYER_NODES = len(x_train[0])
N_OUTPUT_LAYER_NODES = len(y_train[0])
N_HIDDEN_LAYERS = 3
n_hidden_layer_nodes = [1000] * N_HIDDEN_LAYERS

# placeholders for explicit values we know we'll be handling: Input & output
# specify shapes for the benefit of runtime validation
x = tf.placeholder(tf.float32, shape=[None, N_INPUT_LAYER_NODES], name='input')
y = tf.placeholder(tf.float32, shape=[None, N_OUTPUT_LAYER_NODES], name='output')


def sum_(x_, w, b):
    """
    y = X * w + b
    :param x_: input (covariant)
    :param w: weights (contravariant)
    :param b: biases
    :return: output
    """
    return tf.add(tf.matmul(x_, w), b)


def activate_(x_):
    return tf.nn.relu(x_)


def model(data):
    # define tensors
    hidden_layers = []
    n_prev_layer_nodes = N_INPUT_LAYER_NODES
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        n_current_layer_nodes = n_hidden_layer_nodes[hidden_layer_index]
        weights_shape = [n_prev_layer_nodes, n_current_layer_nodes]
        biases_shape = [n_current_layer_nodes]
        hidden_layer = {'weights': tf.Variable(initial_value=tf.random_normal(weights_shape)),
                        'biases': tf.Variable(initial_value=tf.random_normal(biases_shape))}
        hidden_layers.append(hidden_layer)
        n_prev_layer_nodes = n_current_layer_nodes

    output_weights_shape = [n_hidden_layer_nodes[-1], N_OUTPUT_LAYER_NODES]
    output_biases_shape = [N_OUTPUT_LAYER_NODES]
    output_layer = {'weights': tf.Variable(initial_value=tf.random_normal(output_weights_shape)),
                    'biases': tf.Variable(initial_value=tf.random_normal(output_biases_shape))}

    # define operations
    layers = []
    prev_layer = data
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        current_layer = hidden_layers[hidden_layer_index]
        hidden_layer = activate_(sum_(prev_layer, current_layer['weights'], current_layer['biases']))
        layers.append(hidden_layer)
        prev_layer = layers[hidden_layer_index]

    output = sum_(prev_layer, output_layer['weights'], output_layer['biases'])
    return output


def mean_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def run(data, labels):
    prediction = model(data)
    cost = mean_cross_entropy(labels, prediction)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(cost, optimizer, sess)
        test(prediction)


def train(cost, optimizer, sess, max_epochs=10):
    for epoch in range(max_epochs):
        epoch_cost = 0

        total_batches = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
        for batch in range(total_batches):
            start = batch * BATCH_SIZE
            end = min((batch + 1) * BATCH_SIZE, NUM_TRAIN_SAMPLES)
            batch_x, batch_y = array(x_train[start: end]), array(y_train[start: end])
            batch_feed = {x: batch_x, y: batch_y}  # note placeholder keys
            _, batch_cost = sess.run([optimizer, cost], feed_dict=batch_feed)  # optimizer returns None
            # info("Batch %d completed out of %d batches, cost: %g" % (batch, total_batches, batch_cost))
            epoch_cost += batch_cost

        info("Epoch %d completed out of %d epochs, cost: %g " % (epoch + 1, max_epochs, epoch_cost))


def test(prediction):
    test_data = {x: x_test, y: y_test}  # note placeholder keys
    correct_ones = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_ones, tf.float32))
    info("Accuracy: %g" % accuracy.eval(test_data))


run(x, y)
