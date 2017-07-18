"""
Sentiment 140 data set

Training data format:
The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
"""

from logging import basicConfig, DEBUG, info, debug
from os import rename
from os.path import sep, join as path_join
from sys import stdout

from numpy import array
from tensorflow import placeholder, float32, expand_dims, add, matmul, Variable, random_normal, reduce_mean, Session, \
    global_variables_initializer, argmax, cast
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import get_timestamped_export_dir
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits, relu
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.loader import load
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.training.adam import AdamOptimizer

from create_sentiment_featuresets import DATA_DIR
from sentiment_features_large_data import unpickle_processed_data, pickle_processed_data, generate_design_matrix, \
    get_paths_and_lexicon, max_lines

basicConfig(level=DEBUG, stream=stdout)


def get_data(retries=2):
    try:
        if retries == 0:
            raise FileNotFoundError()
        return unpickle_processed_data()
    except FileNotFoundError:
        pickle_processed_data()
        return get_data(retries - 1)


# x_train, y_train, x_test, y_test = get_data()

train_filepath, test_path, lexicon = get_paths_and_lexicon()
first_example = generate_design_matrix(train_filepath, lexicon).__next__()
first_features, first_label = first_example

# Online training
BATCH_SIZE = 1
info(BATCH_SIZE)

N_INPUT_LAYER_NODES = len(first_features)
N_OUTPUT_LAYER_NODES = len(first_label)
N_HIDDEN_LAYERS = 3
n_hidden_layer_nodes = [1000] * N_HIDDEN_LAYERS

# placeholders for explicit values we know we'll be handling: Input & output
# specify shapes for the benefit of runtime validation
# single sample each, for x & y
x = placeholder(float32, shape=[N_INPUT_LAYER_NODES], name='input')
y = placeholder(float32, shape=[N_OUTPUT_LAYER_NODES], name='output')


def sum_(x_, w, b):
    """
    y = X * w + b
    :param x_: input (covariant)
    :param w: weights (contravariant)
    :param b: biases
    :return: output
    """
    non_vector_x = x_ if len(x_.shape) > 1 else expand_dims(x_, 0)
    return add(matmul(non_vector_x, w), b)


def activate_(x_):
    return relu(x_)


def model(data):
    # define tensors
    hidden_layers = []
    n_prev_layer_nodes = N_INPUT_LAYER_NODES
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        n_current_layer_nodes = n_hidden_layer_nodes[hidden_layer_index]
        weights_shape = [n_prev_layer_nodes, n_current_layer_nodes]
        biases_shape = [n_current_layer_nodes]
        hidden_layer = {'weights': Variable(initial_value=random_normal(weights_shape)),
                        'biases': Variable(initial_value=random_normal(biases_shape))}
        hidden_layers.append(hidden_layer)
        n_prev_layer_nodes = n_current_layer_nodes

    output_weights_shape = [n_hidden_layer_nodes[-1], N_OUTPUT_LAYER_NODES]
    output_biases_shape = [N_OUTPUT_LAYER_NODES]
    output_layer = {'weights': Variable(initial_value=random_normal(output_weights_shape)),
                    'biases': Variable(initial_value=random_normal(output_biases_shape))}

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
    return reduce_mean(softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def run(data, labels):
    prediction = model(data)
    cost = mean_cross_entropy(labels, prediction)
    optimizer = AdamOptimizer().minimize(cost)

    with Session() as sess:
        sess.run(global_variables_initializer())
        train(sess, cost, optimizer, max_lines__=int(1e2), max_epochs=1)
        # load_or_train_model(sess, cost, optimizer)
        test(prediction)


def load_or_train_model(sess, cost, optimizer):
    export_dir = path_join(DATA_DIR, "model_backup")
    if maybe_saved_model_directory(export_dir):
        meta_graph_def = load(sess, SERVING, export_dir)
        debug(meta_graph_def)
    else:
        train(sess, cost, optimizer)
        saved_model_path = save_served_model(sess)
        saved_model_dir = saved_model_path.split(sep)[:-1]
        info(saved_model_dir)
        rename(saved_model_dir, export_dir)


def save_served_model(sess):
    """
    estimator.Estimator.export_savedmodel()
    :param sess:
    :return:
    """
    ts_export_dir = get_timestamped_export_dir(DATA_DIR)
    builder = SavedModelBuilder(ts_export_dir)
    builder.add_meta_graph_and_variables(sess, tags=[SERVING])
    return builder.save()


def train(sess, cost, optimizer, max_lines__=max_lines, max_epochs=10):
    for epoch in range(max_epochs):
        epoch_cost = 0

        for sample in generate_design_matrix(train_filepath, lexicon, max_lines_=max_lines__):
            batch_x, batch_y = array(sample[0]), array(sample[1])
            batch_feed = {x: batch_x, y: batch_y}  # note placeholder keys
            _, batch_cost = sess.run([optimizer, cost], feed_dict=batch_feed)  # optimizer returns None
            # info("Batch %d completed out of %d batches, cost: %g" % (batch, total_batches, batch_cost))
            epoch_cost += batch_cost

        info("Epoch %d completed out of %d epochs, cost: %g " % (epoch + 1, max_epochs, epoch_cost))


def test(prediction):
    sum_accuracy = 0
    n_samples = 0
    for sample in generate_design_matrix(test_path, lexicon):
        x_test, y_test = sample
        debug(x_test.nonzero())
        test_data = {x: x_test, y: y_test}  # note placeholder keys
        is_correct = argmax(prediction) == argmax(y)
        accuracy = reduce_mean(cast(1 if is_correct else 0, float32))
        sum_accuracy += accuracy.eval(test_data)
        n_samples += 1
    info("Accuracy: %g" % (sum_accuracy / n_samples))


run(x, y)
