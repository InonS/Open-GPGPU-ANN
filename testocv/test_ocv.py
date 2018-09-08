# utf-8

"""
1. https://github.com/opencv/opencv/blob/master/samples/cpp/neural_network.cpp
2. Detailed example: https://github.com/arnaudgelas/OpenCVExamples/blob/master/NeuralNetwork/NeuralNetwork.cpp
3. https://github.com/opencv/opencv/tree/master/samples/dnn
"""

from logging import DEBUG, basicConfig, debug, info, warning
from os.path import extsep, join as path_join, pardir
from sys import stdout
from time import perf_counter
from timeit import timeit
from typing import List, Tuple

from PIL.Image import Image
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, dnn_Net, getTickFrequency, haveOpenVX, ml_ANN_MLP, rectangle, \
    setUseOpenVX, setUseOptimized, useOpenVX, useOptimized
from cv2.dnn import DNN_BACKEND_HALIDE, DNN_TARGET_OPENCL, blobFromImage, readNetFromTensorflow
from cv2.ipp import getIppVersion, useIPP, useIPP_NE
from cv2.ml import ANN_MLP_BACKPROP, ANN_MLP_SIGMOID_SYM, ANN_MLP_create, ROW_SAMPLE, TrainData_create
from cv2.ocl import finish, haveAmdBlas, haveAmdFft, haveOpenCL, setUseOpenCL, useOpenCL
from numpy import argmax, float32, int32, ndarray, ones, reshape, sort, uint16, uint64, uint8, zeros
from pandas import unique
from sklearn.datasets import load_digits
from sklearn.metrics import log_loss
from tensorflow import GraphDef, Session
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.platform.gfile import FastGFile
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.training.saver import Saver
from tqdm import tqdm, trange


def main():
    ocl_setup()
    ovx_setup()
    # TODO ogl
    ipp_setup()
    optimized_setup()

    x_train, y_train = get_data()
    x_train = x_train.astype(float32)
    mlp: ml_ANN_MLP = train_mlp(x_train, list(y_train.tolist()))
    time_inference(mlp, x_train)

    model_name = "mobilenet_v2_1.0_96"
    logdir = path_join(pardir, "data", model_name)
    frozen_graph = extsep.join(('_'.join((model_name, "frozen")), "pb"))
    net = read_net_from_tf(model_name, logdir, frozen_graph)

    sample = x_train[0]
    width = uint16(8)
    height = uint16(sample.shape[0] / width)
    dnn_predict(net, (width, height), sample, sort(unique(y_train)))


def ocl_setup():
    have_ocl = haveOpenCL()
    info("have_ocl = %s" % have_ocl)
    if have_ocl:
        if not useOpenCL():
            setUseOpenCL(True)
        info("useOpenCL = %s" % useOpenCL())
    info("haveAmdBlas() = %s" % haveAmdBlas())
    info("haveAmdFft() = %s" % haveAmdFft())


def ovx_setup():
    have_ovx = haveOpenVX()
    info("have_ovx = %s" % have_ovx)
    if have_ovx:
        if not useOpenVX():
            setUseOpenVX(True)
        info("useOpenVX() = %s" % useOpenVX())


def ipp_setup():
    """
    Intel Integrated Performance Primitives (IPP)
    """
    info("getIppVersion = %s" % getIppVersion())
    info("useIPP = %s" % useIPP())
    info("useIPP_NE = %s" % useIPP_NE())


def optimized_setup():
    """
    SSE2, NEON, etc.
    """
    if not useOptimized():
        setUseOptimized(True)
    info("useOptimized() = %s" % useOptimized())


def get_data() -> Tuple[ndarray, ndarray]:
    data, target = load_digits(return_X_y=True)
    return data, target


def train_mlp(training_data: ndarray, index: List[uint8]):
    assert training_data.dtype == float32

    n_layers = 3
    layer_sizes = zeros((n_layers, 1), dtype=uint8)  # UMat(3, 1, CV_32SC1).get()

    input_dim = training_data.shape[1]  # 1 << 3
    layer_sizes[0] = input_dim

    hidden_neurons = 1 << 7
    layer_sizes[1] = hidden_neurons

    output_dim = max(index) + 1  # 12
    layer_sizes[2] = output_dim

    mlp: ml_ANN_MLP = ANN_MLP_create()
    mlp.setLayerSizes(layer_sizes)
    mlp.setActivationFunction(ANN_MLP_SIGMOID_SYM, 1, 1)

    n_max_iter = uint8(1e+3)
    max_epsilon = 1e-5
    criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, n_max_iter, max_epsilon)
    mlp.setTermCriteria(criteria)

    mlp.setTrainMethod(ANN_MLP_BACKPROP, 0.1, 0.1)

    train_classes = poker(index, output_dim, training_data)

    koker(mlp, train_classes, training_data)

    if mlp.isTrained():
        predict_ones_vector(mlp, train_classes, training_data)
        predict_validation(mlp, training_data, train_classes)
        mlp.save("neural_network.xml")
    else:
        warning("training failed!")

    return mlp


def poker(index: List[uint8], output_dim, training_data):
    """
    one-hot encoding
    """
    train_classes = zeros((len(training_data), output_dim),
                          dtype=float32)  # UMat(training_data.rows, output_neurons, CV_32FC1).get()
    for i in trange(len(train_classes), desc=None, file=stdout, mininterval=2, unit='sample'):
        train_classes[i, index[i]] = 1
        # debug("Row of train_class: {}".format(train_classes[i]))
        # debug("Row of train_data: {}".format(training_data[i]))
    return train_classes


def koker(mlp: ml_ANN_MLP, train_classes, training_data):
    """
    train
    """
    assert training_data.dtype == float32 or training_data.dtype == int32  # CV_32S (type 4) or CV_32F (type 5)
    assert train_classes.dtype == float32

    td = TrainData_create(training_data, ROW_SAMPLE, train_classes)
    t = timeit("mlp.train(td)", number=1, globals={"mlp": mlp, "td": td})
    info("Training Done (took %f seconds, %f ms per sample)" % (t, 1e3 * t / len(training_data)))


def predict_ones_vector(mlp: ml_ANN_MLP, train_classes, training_data):
    assert training_data.dtype == int32 or training_data.dtype == float32
    assert training_data.shape[1] == mlp.getLayerSizes()[0]

    result = zeros(train_classes.shape[1], dtype=float32)
    assert result.dtype == int32 or result.dtype == float32

    sample = ones((1, training_data.shape[1]), training_data.dtype)
    assert sample.dtype == int32 or sample.dtype == float32
    assert sample.shape[1] == mlp.getLayerSizes()[0]

    predicted, result = mlp.predict(sample)
    info("One-vector prediction {}".format("failed!" if not predicted else ": {}".format(result)))
    return result


def predict_validation(mlp: ml_ANN_MLP, training_data, train_classes):
    predicted = zeros(train_classes.shape[1], dtype=float32)
    assert predicted.dtype == int32 or predicted.dtype == float32

    n_correct = 0
    ll = 0
    for i, sample in tqdm(enumerate(training_data), desc="predict validation", file=stdout, mininterval=2):
        sample = reshape(sample, (1, len(sample)))

        assert sample.dtype == int32 or sample.dtype == float32
        assert sample.shape[1] == mlp.getLayerSizes()[0]

        predicted, result = mlp.predict(sample)
        predicted = uint64(predicted)
        expected = train_classes[i]
        ll += log_loss(reshape(expected, (1, len(expected))), result)
        is_correct = predicted == argmax(expected)  # and allclose(predicted, expected, rtol=5e-2, atol=5e-2)
        n_correct += 1 if is_correct else 0
        # debug("Predict training data ({}): {}".format(is_correct, predicted))
    info("accuracy = %f, log loss per sample = %f" % (
        n_correct / len(training_data), ll / len(training_data)))  # https://en.wikipedia.org/wiki/Confusion_matrix


def time_inference(mlp: ml_ANN_MLP, samples, use_timeit=False):
    if use_timeit:
        results = None
        t = timeit("_, results = mlp.predict(samples)", number=1,
                   globals={"mlp": mlp, "samples": samples, "results": results})
    else:
        start = perf_counter()
        _, results = mlp.predict(samples)
        t = perf_counter() - start

    info("Inference Done (took %f seconds, %f ms per sample)" % (t, 1e3 * t / len(samples)))


def tf_freeze_graph(sess: Session, saver: Saver, model_name: str, logdir: str,
                    model_suffix: str = "pb", conf_suffix: str = "pbtxt", checkpoint_suffix: str = "ckpt"):
    """
    python freeze_graph.py --output_graph =./pbs/frozenGraph.pb --output_node_names=genderOut,ageOut --input_binary=true
    """
    input_conf = write_graph(sess.graph_def, logdir, extsep.join('_'.join((model_name, "graphDef")), conf_suffix))

    input_binary = True
    input_graph = write_graph(sess.graph_def, logdir, extsep.join('_'.join((model_name, "graphDef")), model_suffix),
                              as_text=False)

    input_checkpoint = saver.save(sess, path_join(logdir, extsep.join(model_name, checkpoint_suffix)))

    output_node_names = ""

    output_graph = path_join(logdir, extsep.join('_'.join((model_name, "frozenGraphDef")), model_suffix))
    freeze_graph(input_graph, "", input_binary, input_checkpoint, output_node_names, "save/restore_all", "save/Const:0",
                 output_graph, True, "")
    return output_graph


def optimize_frozen_graph(logdir: str, frozen_graph: str):
    with FastGFile(path_join(logdir, frozen_graph), mode='rb') as frozen_graph_file:
        frozen_graph_def = GraphDef()
        frozen_graph_def.ParseFromString(frozen_graph_file.read())
        optimized_frozen_graph_def = optimize_for_inference(frozen_graph_def, ["Reshape"], ["softmax"], [])
        optimized_frozen_graph_as_bytes = optimized_frozen_graph_def.SerializeToString()

    optimized_frozen_graph = frozen_graph.replace("_frozen", "_frozen_optimized")
    optimized_frozen_graph_path = path_join(logdir, optimized_frozen_graph)
    with open(optimized_frozen_graph_path, mode='wb') as optimized_frozen_graph_file:
        optimized_frozen_graph_file.write(optimized_frozen_graph_as_bytes)

    return optimized_frozen_graph_path


def read_net_from_tf(model_name: str, logdir: str, frozen_graph: str = None):
    if frozen_graph is None:
        with Session as sess:
            frozen_graph = tf_freeze_graph(sess, Saver(), model_name, logdir)

    debug(frozen_graph)

    graph_path = path_join(logdir, frozen_graph)  # optimize_frozen_graph(logdir, frozen_graph)

    net: dnn_Net = readNetFromTensorflow(graph_path)
    net.setPreferableBackend(DNN_BACKEND_HALIDE)
    net.setPreferableTarget(DNN_TARGET_OPENCL)
    return net


def dnn_predict(net: dnn_Net, input_shape, frame: Image, classes):
    """
    https://github.com/opencv/opencv/tree/master/samples/dnn

    https://github.com/tensorflow/models/tree/master/research
    e.g. object_detection (https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API), TF-slim

    https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
    """
    blob = blobFromImage(frame, size=input_shape)

    # Run the model
    net.setInput(blob)
    out = net.forward()

    # Single object classification: Class with the highest score
    flat_out = out.flatten()
    class_id = argmax(flat_out)
    confidence = flat_out[class_id]

    # Predicted class
    info('%s: %.4f' % (classes[class_id] if classes else 'Class #%d' % class_id, confidence))

    # Multiple object detection:
    bbox_color = (0, 255, 0)
    threshold = 0.5  # 0.3
    for detection in out[0, 0, :, :]:
        score = float(detection[2])
        if score > threshold:
            left = detection[3] * frame.width
            top = detection[4] * frame.height

            right = detection[5] * frame.width
            bottom = detection[6] * frame.height

            tl = (uint16(left), uint16(top))
            br = (uint16(right), uint16(bottom))

            rectangle(frame, tl, br, bbox_color)

    # Efficiency information
    t, _ = net.getPerfProfile()
    info('Inference time: %.2f ms' % (t * 1000.0 / getTickFrequency()))


if __name__ == '__main__':
    basicConfig(stream=stdout, level=DEBUG)
    main()
    finish()
