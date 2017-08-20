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

from os import rename
from os.path import join as path_join, sep

from numpy import array
from tensorflow import Session, Variable, add, argmax, cast, expand_dims, float32, global_variables_initializer, \
    matmul, placeholder, random_normal, reduce_mean
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import get_timestamped_export_dir
from tensorflow.python.ops.nn_ops import relu, softmax_cross_entropy_with_logits
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.loader import load
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.summary.summary import histogram, merge_all, scalar
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.tensorboard_logging import DEBUG, debug, info, set_summary_writer, set_verbosity

from create_sentiment_featuresets import DATA_DIR
from sentiment_features_large_data import MAX_LINES, generate_design_matrix, get_paths_and_lexicon, \
    pickle_processed_data, unpickle_processed_data

LOGDIR = "log"


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

N_INPUT_LAYER_NODES = int()
N_OUTPUT_LAYER_NODES = int()
N_HIDDEN_LAYERS = int()
N_HIDDEN_LAYER_NODES = list()


def define_model_params(n_hidden_layers=3, n_nodes_per_hidden_layer=1000):
    global N_INPUT_LAYER_NODES, N_OUTPUT_LAYER_NODES, N_HIDDEN_LAYERS, N_HIDDEN_LAYER_NODES
    first_example = generate_design_matrix(train_filepath, lexicon).__next__()
    first_features, first_label = first_example

    N_INPUT_LAYER_NODES = len(first_features)
    N_OUTPUT_LAYER_NODES = len(first_label)

    N_HIDDEN_LAYERS = n_hidden_layers
    N_HIDDEN_LAYER_NODES = [n_nodes_per_hidden_layer] * N_HIDDEN_LAYERS


x = placeholder(float32, name='input')
y = placeholder(float32, name='output')


def define_global_placeholders():
    # placeholders for explicit values we know we'll be handling: Input & output
    # specify shapes for the benefit of runtime validation
    # single sample each, for x & y
    global x, y
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
    hidden_layers, output_layer = define_tensors()
    return define_operations(data, hidden_layers, output_layer)


def define_operations(data, hidden_layers, output_layer):
    layers = []
    prev_layer = data
    histogram("input_layer", data)
    histogram("preactivation_layer_%d", data)
    histogram("activation_layer_%d", data)
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        current_layer = hidden_layers[hidden_layer_index]
        preactivation = sum_(prev_layer, current_layer['weights'], current_layer['biases'])
        hidden_layer = activate_(preactivation)
        layers.append(hidden_layer)
        prev_layer = layers[hidden_layer_index]
    output = sum_(prev_layer, output_layer['weights'], output_layer['biases'])
    return output


def define_tensors():
    hidden_layers = define_hidden_layer_tensors()
    output_layer = define_output_layer_tensors()
    return hidden_layers, output_layer


def define_output_layer_tensors():
    output_weights_shape = [N_HIDDEN_LAYER_NODES[-1], N_OUTPUT_LAYER_NODES]
    output_biases_shape = [N_OUTPUT_LAYER_NODES]
    output_layer = {'weights': Variable(initial_value=random_normal(output_weights_shape)),
                    'biases': Variable(initial_value=random_normal(output_biases_shape))}
    histogram("output_layer_weights", output_layer['weights'])
    histogram("output_layer_biases", output_layer['biases'])
    return output_layer


def define_hidden_layer_tensors():
    hidden_layers = []
    n_prev_layer_nodes = N_INPUT_LAYER_NODES
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        n_current_layer_nodes = N_HIDDEN_LAYER_NODES[hidden_layer_index]
        weights_shape = [n_prev_layer_nodes, n_current_layer_nodes]
        biases_shape = [n_current_layer_nodes]
        hidden_layer = {'weights': Variable(initial_value=random_normal(weights_shape)),
                        'biases': Variable(initial_value=random_normal(biases_shape))}
        hidden_layers.append(hidden_layer)
        histogram("layer_%d_weights" % hidden_layer_index, hidden_layer['weights'])
        histogram("layer_%d_biases" % hidden_layer_index, hidden_layer['biases'])
        n_prev_layer_nodes = n_current_layer_nodes
    return hidden_layers


def mean_cross_entropy(labels, logits):
    return reduce_mean(softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def run(data, labels, max_lines_=None, max_epochs=None):
    latest_predictions = model(data)
    cost = mean_cross_entropy(labels, latest_predictions)
    optimizer = AdamOptimizer(learning_rate=0.01).minimize(cost)

    # TODO Consider MonitoredTrainingSession or SessionManager
    with Session() as sess:
        sess.run(global_variables_initializer())
        train(sess, cost, optimizer, max_lines_=max_lines_, max_epochs=max_epochs)
        # load_or_train_model(sess, cost, optimizer, max_lines_=max_lines_, max_epochs=max_epochs)
        test(latest_predictions)


def load_or_train_model(sess, cost, optimizer, max_lines_=MAX_LINES, max_epochs=None):
    """
    TODO Make into a decorator
    """
    export_dir = path_join(DATA_DIR, "model_backup")
    if maybe_saved_model_directory(export_dir):
        meta_graph_def = load(sess, SERVING, export_dir)
        info("meta-graph def = {}.".format(meta_graph_def))
    else:
        train(sess, cost, optimizer, max_lines_=max_lines_, max_epochs=max_epochs)
        saved_model_path = save_served_model(sess)
        saved_model_dir = saved_model_path.split(sep)[:-1]
        info("saved_model_dir = %s" % saved_model_dir)
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


def train(sess, cost, optimizer, max_lines_=MAX_LINES, max_epochs=None):
    """
    ~ 80 training samples / minute = 4/31 samples / second = 4800 samples / hour = 1.152e5 samples / day
    1.6e6 training samples = 13 8/9 days for one-shot (max_epochs = 1).

    (4800 samples / hour) * 5 hours / 10 epochs = 2400 samples
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    """
    summary_writer = FileWriter(path_join(LOGDIR, "train"), graph=sess.graph)
    set_summary_writer(summary_writer)
    set_verbosity(DEBUG)

    # for sample_text in batch_x:
    #     text_summary("input_text", [lexicon[word] for word in sample_text])
    epoch = 0
    scalar('epoch', epoch)
    epoch_cost = 0
    scalar('epoch_cost', epoch_cost)
    batch_cost = 0
    scalar('batch_cost', batch_cost)
    merged_summaries = merge_all()

    if max_epochs is None:
        max_epochs = 1
    for epoch in range(max_epochs):
        epoch_cost = 0

        for sample in generate_design_matrix(train_filepath, lexicon, max_lines_=max_lines_):
            batch_x, batch_y = array(sample[0]), array(sample[1])
            batch_feed = {x: batch_x, y: batch_y}  # note placeholder keys
            _, batch_cost, summary = sess.run(
                [optimizer, cost, merged_summaries], feed_dict=batch_feed)  # optimizer returns None
            # info("Batch %d completed out of %d batches, cost: %g" % (batch, total_batches, batch_cost))
            epoch_cost += batch_cost
            summary_writer.add_summary(summary)

        info("Epoch %d completed out of %d epochs, cost: %g " % (epoch + 1, max_epochs, epoch_cost))


def test(prediction):
    summary_writer = FileWriter(path_join(LOGDIR, "test"))
    set_summary_writer(summary_writer)
    set_verbosity(DEBUG)

    n_samples = 0
    scalar('test_samples', n_samples)
    sum_accuracy = 0
    running_accuracy = 0
    scalar('running_accuracy', running_accuracy)
    merged_summaries = merge_all()

    for sample in generate_design_matrix(test_path, lexicon):
        x_test, y_test = sample
        debug("test sample: # features = {}, labels = {}".format(len(x_test), y_test))
        # debug("nonzero elements in test set: {}".format(x_test.nonzero()))
        test_data = {x: x_test, y: y_test}  # note placeholder keys
        is_correct = argmax(prediction) == argmax(y)
        accuracy = reduce_mean(cast(1 if is_correct else 0, float32))
        sample_accuracy = accuracy.eval(test_data)
        # summary = sess.run(merged_summaries)
        n_samples += 1
        sum_accuracy += sample_accuracy
        running_accuracy = sum_accuracy / n_samples
        # summary_writer.add_summary(summary)

    info("Accuracy: %g" % running_accuracy)


def main():
    define_model_params()
    define_global_placeholders()
    run(x, y, max_lines_=int(1e1))  # , max_lines_=int(2.5e3), max_epochs=10)


if __name__ == '__main__':
    main()
