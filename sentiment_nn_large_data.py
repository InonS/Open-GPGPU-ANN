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
from tensorflow import Session, Variable, argmax, cast, constant, expand_dims, float32, global_variables_initializer, \
    name_scope, placeholder, random_normal, reduce_mean, string
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import get_timestamped_export_dir
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.python.ops.nn_ops import relu, softmax_cross_entropy_with_logits, xw_plus_b
from tensorflow.python.profiler.tfprof_logger import write_op_log
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.loader import load
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.summary.summary import histogram, merge, merge_all, scalar
from tensorflow.python.summary.text_summary import text_summary
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.tensorboard_logging import DEBUG, debug, info, set_summary_writer, set_verbosity

from create_sentiment_featuresets import DATA_DIR
from sentiment_features_large_data import MAX_LINES, generate_design_matrix, get_paths_and_lexicon

# logs locations
LOGDIR = str(get_timestamped_export_dir("log"))[2:-2]
TRAIN_DIR = path_join(LOGDIR, "train")
TEST_DIR = path_join(LOGDIR, "test")

# data locations
train_filepath, test_path, lexicon = get_paths_and_lexicon()

# sample data
first_example = generate_design_matrix(train_filepath, lexicon).__next__()
first_features, first_label = first_example

# network boundaries
N_INPUT_LAYER_NODES = len(first_features)
N_OUTPUT_LAYER_NODES = len(first_label)

# hidden layer
N_HIDDEN_LAYERS = 3
N_NODES_PER_HIDDEN_LAYER = 1000
N_HIDDEN_LAYER_NODES = [N_NODES_PER_HIDDEN_LAYER] * N_HIDDEN_LAYERS

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
    x_rank_gt_1 = x_ if len(x_.shape) > 1 else expand_dims(x_, 0)
    return xw_plus_b(x_rank_gt_1, w, b)


def activate_(x_):
    return relu(x_)


def model(data):
    hidden_layers, output_layer = define_tensors()
    return define_operations(data, hidden_layers, output_layer)


def define_operations(data, hidden_layers, output_layer):
    layers = []
    prev_layer = data
    histogram("input_layer", data)
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        with name_scope("layer_%d_ops" % hidden_layer_index):
            current_layer = hidden_layers[hidden_layer_index]
            preactivation = sum_(prev_layer, current_layer['weights'], current_layer['biases'])
            histogram("preactivation", preactivation)
            hidden_layer = activate_(preactivation)
            histogram("activation", hidden_layer)
            layers.append(hidden_layer)
            prev_layer = layers[hidden_layer_index]
    with name_scope("output_layer_ops"):
        output = sum_(prev_layer, output_layer['weights'], output_layer['biases'])
    return output


def define_tensors():
    hidden_layers = define_hidden_layer_tensors()
    with name_scope("output_layer_vars"):
        output_layer = define_output_layer_tensors()
    return hidden_layers, output_layer


def define_output_layer_tensors():
    output_weights_shape = [N_HIDDEN_LAYER_NODES[-1], N_OUTPUT_LAYER_NODES]
    output_biases_shape = [N_OUTPUT_LAYER_NODES]
    output_layer = {
        'weights': Variable(initial_value=random_normal(output_weights_shape), name="output_weights", dtype=float32,
                            expected_shape=output_weights_shape),  # collections=[GraphKeys.WEIGHTS]
        'biases': Variable(initial_value=random_normal(output_biases_shape), name="output_biases", dtype=float32,
                           expected_shape=output_biases_shape)}  # collections=[GraphKeys.BIASES]
    histogram("output_layer_weights", output_layer['weights'])
    histogram("output_layer_biases", output_layer['biases'])
    return output_layer


def define_hidden_layer_tensors():
    hidden_layers = []
    n_prev_layer_nodes = N_INPUT_LAYER_NODES
    for hidden_layer_index in range(N_HIDDEN_LAYERS):
        with name_scope("layer_%d_vars" % hidden_layer_index):
            n_current_layer_nodes = N_HIDDEN_LAYER_NODES[hidden_layer_index]
            weights_shape = [n_prev_layer_nodes, n_current_layer_nodes]
            biases_shape = [n_current_layer_nodes]
            hidden_layer = {
                'weights': Variable(initial_value=random_normal(weights_shape),
                                    name="layer_%d_weights" % hidden_layer_index, dtype=float32,
                                    expected_shape=weights_shape),  # collections=[GraphKeys.WEIGHTS]
                'biases': Variable(initial_value=random_normal(biases_shape),
                                   name="layer_%d_biases" % hidden_layer_index, dtype=float32,
                                   expected_shape=biases_shape)}  # collections=[GraphKeys.BIASES]
            hidden_layers.append(hidden_layer)
            histogram("weights", hidden_layer['weights'])
            histogram("biases", hidden_layer['biases'])
        n_prev_layer_nodes = n_current_layer_nodes

    return hidden_layers


def mean_cross_entropy(labels, logits):
    xent = softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="xent")
    return reduce_mean(xent, name="mean_cross_entropy")


def run(data, labels, max_lines=None, max_epochs=None):
    latest_predictions = model(data)
    optimizer = build_optimizer(labels, latest_predictions)

    # TODO Consider MonitoredTrainingSession or SessionManager
    with Session() as sess:
        sess.run(global_variables_initializer())

        # advise(sess.graph)

        summary_writer = FileWriter(TRAIN_DIR, graph=sess.graph)
        with name_scope("training"):
            write_op_log(sess.graph, TRAIN_DIR)
            train(sess, summary_writer, optimizer, max_lines=max_lines, max_epochs=max_epochs, tb_text_samples=100)
            # load_or_train_model(sess, cost, optimizer, max_lines_=max_lines_, max_epochs=max_epochs)
        summary_writer.flush()
        summary_writer.close()

        summary_writer = FileWriter(TEST_DIR)
        with name_scope("testing"):
            write_op_log(sess.graph, TEST_DIR)
            test(sess, summary_writer, latest_predictions)
        summary_writer.flush()
        summary_writer.close()


def build_optimizer(labels, latest_predictions):
    with name_scope("cost_func"):
        cost = mean_cross_entropy(labels, latest_predictions)
    scalar("cost", cost)
    with name_scope("opt_algo"):
        return AdamOptimizer().minimize(cost)


def load_or_train_model(sess, optimizer, max_lines_=MAX_LINES, max_epochs=None):
    """
    TODO Make into a decorator
    """
    export_dir = path_join(DATA_DIR, "model_backup")
    if maybe_saved_model_directory(export_dir):
        meta_graph_def = load(sess, SERVING, export_dir)
        info("meta-graph def = {}.".format(meta_graph_def))
    else:
        summary_writer = FileWriter(TRAIN_DIR, graph=sess.graph)
        train(sess, summary_writer, optimizer, max_lines=max_lines_, max_epochs=max_epochs)
        summary_writer.flush()
        summary_writer.close()

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


def train(sess, summary_writer, optimizer, max_lines=None, max_epochs=None, tb_text_samples=None):
    """
    ~ 4/3 samples / second = 4800 samples / hour = 1.152e5 samples / day
    1.6e6 training samples = 13 8/9 days for one-shot (max_epochs = 1).

    (4800 samples / hour) * 5 hours / 1 epochs = 24000 samples
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    """
    set_summary_writer(summary_writer)
    set_verbosity(DEBUG)

    debug("Local devices: {}".format(list_local_devices()))

    # text = Variable(initial_value="", dtype=string, name="sample_text")
    # text_summary("input_text", text)
    merged_summaries = None
    # merged_summaries = merge_all()

    global_step = 0
    if tb_text_samples is None:
        tb_text_samples = max_lines / (max_lines + 1)
    else:
        tb_text_samples = min(tb_text_samples, max_lines)
    if max_epochs is None:
        max_epochs = 1
    for epoch in range(max_epochs):
        for sample in generate_design_matrix(train_filepath, lexicon, max_lines=max_lines):
            sample_x, sample_y = array(sample[0]), array(sample[1])

            if epoch == 0 and global_step % (max_lines / tb_text_samples) == 0:
                sample_words = [lexicon[int(word_index)] for word_index in sample_x]
                lexed_sample = ' '.join(sample_words)
                sample_text = constant(lexed_sample, dtype=string, name="sample_text")
                # text.assign(sample_text)
                text_summary_ = text_summary("sample_text", sample_text)
                merged_summaries = merge(
                    [merged_summaries, text_summary_] if merged_summaries is not None else [text_summary_])

            batch_feed = {x: sample_x, y: sample_y}  # note placeholder keys
            summary, _ = sess.run([merged_summaries, optimizer], feed_dict=batch_feed)
            summary_writer.add_summary(summary, global_step=global_step)
            global_step += 1


def build_accuracy(y_true, y_pred):
    with name_scope("evaluation_metrics"):
        argmax_y_pred = argmax(y_pred, axis=1, name="argmax_y_pred")[0]
        debug("y_pred.shape = {}, argmax_y_pred = {}".format(y_pred.shape, argmax_y_pred))
        scalar("argmax_y_pred", argmax_y_pred)

        argmax_y_true = argmax(y_true, axis=0, name="argmax_y_true")
        debug("y_true.shape = {}, argmax_y_true = {}".format(y_true.shape, argmax_y_true))
        scalar("argmax_y_true", argmax_y_true)

        histogram("ground_truth", y_true)
        histogram("prediction", y_pred)

        # correct_classifications = equal(argmax_y_pred, argmax_y_true)
        correct_classifications = argmax_y_pred == argmax_y_true
        # histogram("correct_classification", correct_classifications)
        indicator = cast(1 if correct_classifications else 0, float32, name="indicator")
        histogram("correct_indicator", indicator)
        accuracy = reduce_mean(indicator)
        scalar("accuracy", accuracy)
        return accuracy


def test(sess, summary_writer, prediction):
    set_summary_writer(summary_writer)
    set_verbosity(DEBUG)

    accuracy = build_accuracy(y, prediction)
    # auc = Variable(initial_value=0, name="AUC")
    # scalar("AUC", auc)
    merged_summaries = merge_all()

    overall_accuracy = 0
    global_step = 0
    for sample in generate_design_matrix(test_path, lexicon):
        x_test, y_test = sample
        # debug("nonzero elements in test set: {}".format(x_test.nonzero()))
        test_data = {x: x_test, y: y_test}  # note placeholder keys
        sample_accuracy = accuracy.eval(test_data)
        # if global_step % 10 == 0:
        #     debug("overall_accuracy = %g" % overall_accuracy)
        summary = sess.run(merged_summaries, feed_dict=test_data)
        # _streaming_confusion_matrix_at_thresholds(prediction, y, [0.5, 0.5])
        # auc.assign(streaming_auc(prediction, y > 0.5).eval(test_data))
        summary_writer.add_summary(summary, global_step=global_step)
        global_step += 1
        overall_accuracy += sample_accuracy

    info("overall accuracy = %g" % overall_accuracy)


def main():
    run(x, y, max_lines=int(3e3))


if __name__ == '__main__':
    main()
