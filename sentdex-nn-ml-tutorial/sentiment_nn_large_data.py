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

from math import sqrt
from os import replace
from os.path import join as path_join, sep

from create_sentiment_featuresets import DATA_DIR
from sentiment_features_large_data import generate_design_matrix, get_paths_and_lexicon
from tensorflow import Session, Variable, argmax, constant, equal, expand_dims, float32, global_variables_initializer, \
    name_scope, placeholder, random_normal, reduce_mean, string, to_float
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import get_timestamped_export_dir
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.python.ops.nn_ops import relu, softmax_cross_entropy_with_logits, xw_plus_b
from tensorflow.python.profiler.tfprof_logger import write_op_log
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.loader import load
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
from tensorflow.python.saved_model.tag_constants import SERVING, TRAINING
from tensorflow.python.summary.summary import histogram, merge, merge_all, scalar
from tensorflow.python.summary.text_summary import text_summary
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.learning_rate_decay import exponential_decay
from tensorflow.python.training.tensorboard_logging import DEBUG, debug, info, set_summary_writer, set_verbosity

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


# online: BATCH_SIZE = 1

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
    prev_layer = data  # TODO data = csc_matrix(data) \n prev_layer = SparseTensor(data.indices, data.data, data.shape)
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


def run(data, labels, max_lines_train=None, max_lines_test=None, max_epochs=None):
    latest_predictions = model(data)
    train_op = build_train_step_op(labels, latest_predictions, learning_rate=0.01)

    # TODO Consider MonitoredTrainingSession or SessionManager
    with Session() as sess:
        sess.run(global_variables_initializer())
        load_or_train_model(sess, train_op, max_lines=max_lines_train, max_epochs=max_epochs)
        # info(advise(sess.graph)) # ...\tensorflow\core\profiler\internal\tfprof_code.cc:66]
        # Check failed: trace_root_->name() == trace Different trace root
        test(sess, latest_predictions, max_lines=max_lines_test)


def build_train_step_op(labels, latest_predictions, learning_rate=0.001, global_step=None):
    with name_scope("cost_func"):
        histogram("input_labels", labels)
        histogram("predictions", latest_predictions)
        cost = mean_cross_entropy(labels, latest_predictions)
    scalar("cost", cost)
    with name_scope("opt_algo"):
        if global_step is not None:
            learning_rate = exponential_decay(learning_rate, global_step, int(1e4), 0.96)
        optimizer = AdamOptimizer(learning_rate=learning_rate)
    return optimizer.minimize(cost, global_step=global_step, name="train_step_op")


def load_or_train_model(sess, train_op, max_lines=None, max_epochs=None, tb_text_samples=None, cache=False):
    """
    TODO Make into a decorator
    """
    export_dir = path_join(DATA_DIR, "model_backup")
    if cache and maybe_saved_model_directory(export_dir):
        meta_graph_def = load(sess, SERVING, export_dir)
        info("meta-graph def = {}.".format(meta_graph_def))
    else:
        summary_writer = FileWriter(TRAIN_DIR, graph=sess.graph)
        set_summary_writer(summary_writer)
        set_verbosity(DEBUG)
        debug("Local devices: {}".format(list_local_devices()))
        info("Run Tensorboard with '--logdir log' and access it at localhost:6006 (%s)" % LOGDIR)
        with name_scope("training"):
            write_op_log(sess.graph, TRAIN_DIR)
            train(sess, train_op, summary_writer, max_lines=max_lines, max_epochs=max_epochs,
                  tb_text_samples=tb_text_samples)
        summary_writer.flush()
        summary_writer.close()

        saved_model_path = save_model(sess, is_served=False)
        saved_model_dir = sep.join(saved_model_path.split(sep)[:-1])
        info("saved_model_dir = %s" % saved_model_dir)
        try:
            replace(saved_model_dir, export_dir)
        except PermissionError:
            return


def save_model(sess, is_served=True):
    """
    estimator.Estimator.export_savedmodel()
    """
    ts_export_dir = get_timestamped_export_dir(DATA_DIR)
    builder = SavedModelBuilder(ts_export_dir)
    builder.add_meta_graph_and_variables(sess, tags=[SERVING if is_served else TRAINING])
    save_path_bytes = builder.save()
    return str(save_path_bytes)[2:-1]


def train(sess, train_op, summary_writer, max_lines=None, max_epochs=None, tb_text_samples=None):
    """
    ~ 1 samples / second, ~10 Kb / sample (summary protobufs storage)
    https://www.tensorflow.org/get_started/summaries_and_tensorboard

    cost(step = t) ~ 1e4 * exp(-t/5e3) -> cost(t0 + 5e3) / cost(t0) = 1/e ~ 37%

    See also basic_loops.basic_train_loop
    """
    merged_summaries = merge_all()

    global_step = 0
    if tb_text_samples is None:
        tb_text_samples = 0.5  # tb_text_samples < 1 means text_summary will never be written
    else:
        tb_text_samples = min(tb_text_samples, max_lines)
    if max_epochs is None:
        max_epochs = 1
    for epoch in range(max_epochs):
        for sample in generate_design_matrix(train_filepath, lexicon, max_lines=max_lines):
            sample_x, sample_y = sample[0], sample[1]
            if epoch == 0 and global_step % (max_lines / tb_text_samples) == 0:
                nz = sample_x.nonzero()
                sample_words = [lexicon[word_index] for word_index in nz[0]]
                lexed_sample = ' '.join(sample_words)
                # debug("sample #%d: %s <- %s" % (global_step, str(sample_y), lexed_sample))
                text = constant(lexed_sample, dtype=string, name="input_text_var")
                text_summary_ = text_summary("input_text", text)
                histogram("sample_sentiment", sample_y)
                merged_summaries = merge([merged_summaries, text_summary_])

            feed = {x: sample_x, y: sample_y}  # note placeholder keys
            summary, _ = sess.run([merged_summaries, train_op], feed_dict=feed)
            # debug("y = {}, pred = {}".format(*sess.run([latest_predictions, sample_y], feed_dict=feed)))
            summary_writer.add_summary(summary, global_step=global_step)
            global_step += 1
            # info("global step %d: lines read = %d" % (global_step, line))


def build_accuracy(y_true, y_pred):
    """
    Handles batch samples as well as online
    TODO deprecate in favor of metric_ops.streaming_accuracy()
    """
    with name_scope("evaluation_metrics"):
        histogram("prediction", y_pred)
        y_true_rank_gt_1 = y_true if len(y_true.shape) > 1 else expand_dims(y_true, 0)
        histogram("ground_truth", y_true_rank_gt_1)

        argmax_y_pred = argmax(y_pred, axis=1, name="argmax_y_pred_op")
        argmax_y_true = argmax(y_true_rank_gt_1, axis=1, name="argmax_y_true_op")

        is_correct = equal(argmax_y_pred, argmax_y_true, name="is_correct")
        indicator = to_float(is_correct, name="indicator")
        histogram("correct_indicator", indicator)
        accuracy = reduce_mean(indicator)
        scalar("accuracy", accuracy)
        return accuracy


def test(sess, prediction, max_lines=None):
    summary_writer = FileWriter(TEST_DIR, graph=sess.graph)
    with name_scope("testing"):
        set_summary_writer(summary_writer)
        set_verbosity(DEBUG)

        write_op_log(sess.graph, TEST_DIR)

        accuracy = build_accuracy(y, prediction)
        # confusn_mtx, confusn_mtx_update_op = _streaming_confusion_matrix_at_thresholds(prediction[0], y, [0.5, 0.5])
        # auc_update_op = streaming_auc(prediction, y > 0.5)
        merged_summaries = merge_all()

        # accuracy statistics
        total_accuracy = 0  # assign_moving_average(running_accuracy, accuracy, 1 / max_lines ...)
        total_squared_accuracy = 0

        # global step
        global_step = 0
        # global_step_tensor = get_or_create_global_step(sess.graph)
        # global_step_ = global_step(sess, global_step_tensor)

        for sample in generate_design_matrix(test_path, lexicon):
            x_test, y_test = sample
            test_data = {x: x_test, y: y_test}  # note placeholder keys
            sample_accuracy = accuracy.eval(test_data)
            total_accuracy += sample_accuracy
            total_squared_accuracy += sample_accuracy
            summary = sess.run(merged_summaries, feed_dict=test_data)
            summary_writer.add_summary(summary, global_step=global_step)
            global_step += 1

            if max_lines is not None and global_step == max_lines:
                break

        total_accuracy /= global_step
        total_squared_accuracy /= global_step
        info("Accuracy = %g +/- %g" % (total_accuracy, sqrt(total_squared_accuracy - total_accuracy * total_accuracy)))

    summary_writer.flush()
    summary_writer.close()


def main():
    run(x, y, max_lines_train=3e3)  # 2e4)


if __name__ == '__main__':
    main()
