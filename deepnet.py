"""
1. Build feed-forward net: input -> weight -> hidden layer 1 (activation function) -> weight -> hidden l2 (activation) -> weights -> output
2. Compare output to label with cost (or loss) function (e.g. cross entropy)
3. Minimize cost using optimizer (e.g. ADAM, SGD, AdaGrad, etc.)
4. Propagate the correction back to the weights in the original network: Back-propagation
1 + 2 + 3 + 4 = epoch
repeat until convergence (if not monotonous, use early-stopping)
"""
from logging import basicConfig, DEBUG, debug, info
from sys import stdout, getsizeof
from tempfile import mkdtemp

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

basicConfig(level=DEBUG, stream=stdout)

"""
 1. get MNIST data
    a. 28x28 b/w pixel pictures of handwritten digits 
    b. 50,000 training samples and 5,000 test samples
    c. Multi-class (10 classes: '0', '1', ..., '9') classification
    d. one-hot encoding: '0' = [1, 0, 0, ..., 0] ; '1' = [0, 1, 0, ..., 0] ; ... ; '9' = [0, 0, 0, ..., 1]
"""
tempd = mkdtemp()
debug(tempd)
mnist = input_data.read_data_sets(tempd, one_hot=True)
debug(mnist)
NUM_TRAIN_SAMPLES = mnist.train.num_examples
info(NUM_TRAIN_SAMPLES)

# Out-of-core training: Batches
SAMPLE_SIZE = getsizeof(mnist.train.images[0])
KIBI = 1 << 10
GIBI = KIBI * KIBI * KIBI
RAM_SIZE = 4 * GIBI
BATCH_SIZE = int(RAM_SIZE / SAMPLE_SIZE)
if NUM_TRAIN_SAMPLES <= BATCH_SIZE:
    BATCH_SIZE = int(NUM_TRAIN_SAMPLES / 10)
info(BATCH_SIZE)
"""
2. Build model:
    a. Input layer: 28*28 = 784 float-valued nodes (flattened 2-D picture into a 1-D array)
    b. Output layer: 10 nodes
    c. e.g. 3 hidden layers: e.g. 500 nodes each
"""
N_INPUT_LAYER_NODES = len(mnist.train.images[0])
N_OUTPUT_LAYER_NODES = len(mnist.train.labels[0])
N_HIDDEN_LAYERS = 3
n_hidden_layer_nodes = [500] * N_HIDDEN_LAYERS

# placeholders for explicit values we know we'll be handling: Input & output
# specify shapes for the benefit of runtime validation
x = tf.placeholder(tf.float32, shape=[None, N_INPUT_LAYER_NODES], name='input')
y = tf.placeholder(tf.float32, shape=[None, N_OUTPUT_LAYER_NODES], name='output')


def sigma(x_, w, b):
    """
    y = X * w + b
    :param x_: input (covariant)
    :param w: weights (contravariant)
    :param b: biases
    :return: output
    """
    return tf.add(tf.matmul(x_, w), b)


def activation(x_):
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
        hidden_layer = activation(sigma(prev_layer, current_layer['weights'], current_layer['biases']))
        layers.append(hidden_layer)
        prev_layer = layers[hidden_layer_index]

    return sigma(prev_layer, output_layer['weights'], output_layer['biases'])


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
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_feed = {x: batch_x, y: batch_y}  # note placeholder keys
            _, batch_cost = sess.run([optimizer, cost], feed_dict=batch_feed)  # optimizer returns None
            # info("Batch %d completed out of %d batches, cost: %g" % (batch, total_batches, batch_cost))
            epoch_cost += batch_cost

        info("Epoch %d completed out of %d epochs, cost: %g " % (epoch, max_epochs, epoch_cost))


def test(prediction):
    test_samples = mnist.test
    test_data = {x: test_samples.images, y: test_samples.labels}  # note placeholder keys
    correct_ones = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_ones, tf.float32))
    info("Accuracy: %g" % accuracy.eval(test_data))


run(x, y)
