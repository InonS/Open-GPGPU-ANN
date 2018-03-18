# utf-8

"""
1. https://github.com/opencv/opencv/blob/master/samples/cpp/neural_network.cpp
2. Detailed example: https://github.com/arnaudgelas/OpenCVExamples/blob/master/NeuralNetwork/NeuralNetwork.cpp
3. https://github.com/opencv/opencv/tree/master/samples/dnn
"""

from logging import INFO, basicConfig, debug, info
from sys import stdout
from timeit import timeit
from typing import Iterable, Tuple, List

from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, haveOpenVX, ml_ANN_MLP, setUseOpenVX, setUseOptimized, \
    useOpenVX, useOptimized
from cv2.ipp import getIppVersion, useIPP, useIPP_NE
from cv2.ml import ANN_MLP_BACKPROP, ANN_MLP_SIGMOID_SYM, ANN_MLP_create, ROW_SAMPLE, TrainData_create
from cv2.ocl import finish, haveAmdBlas, haveAmdFft, haveOpenCL, setUseOpenCL, useOpenCL
from numpy import float32, int32, ndarray, uint8, zeros
from sklearn.datasets import load_digits
from tqdm import trange


def main():
    ocl_setup()
    ovx_setup()
    # TODO ogl
    ipp_setup()
    optimized_setup()

    x_train, y_train = get_data()
    x_train = x_train.astype(float32)
    train_mlp(x_train, y_train.tolist())  # TODO readNetFromTensorflow()


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

    criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, uint8(1e+3), 1e-5)
    mlp.setTermCriteria(criteria)

    mlp.setTrainMethod(ANN_MLP_BACKPROP, 0.1, 0.1)  # mlp.setTrainMethod(ANN_MLP_SIGMOID_SYM)

    train_classes = poker(index, output_dim, training_data)

    koker(mlp, train_classes, training_data)

    mlp.save("neural_network.xml")


def koker(mlp, train_classes, training_data):
    assert training_data.dtype == float32 or training_data.dtype == int32  # CV_32S (type 4) or CV_32F (type 5)
    assert train_classes.dtype == float32

    td = TrainData_create(training_data, ROW_SAMPLE, train_classes)
    t = timeit("mlp.train(td)", number=1, globals={"mlp": mlp, "td": td})
    print("Training Done (took %f seconds, %f ms per sample)" % (t, 1e3 * t / len(training_data)))


def poker(index: List[uint8], output_dim, training_data):
    """
    one-hot encoding
    """
    train_classes = zeros((len(training_data), output_dim),
                          dtype=float32)  # UMat(training_data.rows, output_neurons, CV_32FC1).get()
    for i in trange(len(train_classes), desc=None, file=stdout, mininterval=2, unit='sample'):
        train_classes[i, index[i]] = 1
        debug("Row of train_class: {}".format(train_classes[i]))
        debug("Row of train_data: {}".format(training_data[i]))
    return train_classes


if __name__ == '__main__':
    basicConfig(stream=stdout, level=INFO)
    main()
    finish()
