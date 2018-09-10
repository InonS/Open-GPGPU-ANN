# TODO https://github.com/BVLC/caffe/tree/opencl
# TODO https://github.com/01org/caffe/wiki/clCaffe
# TODO https://github.com/BVLC/caffe/issues/5099

from logging import DEBUG, basicConfig, info

import caffe


def main():
    info(caffe.__version__)


if __name__ == '__main__':
    basicConfig(level=DEBUG)
    main()
