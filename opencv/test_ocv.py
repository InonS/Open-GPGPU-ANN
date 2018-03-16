from logging import DEBUG, basicConfig, info
from sys import stdout

from cv2 import haveOpenVX, setUseOpenVX, setUseOptimized, useOpenVX, useOptimized


def main():
    have_ovx = haveOpenVX()
    info(have_ovx)
    if have_ovx:
        if not useOpenVX():
            setUseOpenVX(True)
        info(useOpenVX())

    if not useOptimized():
        setUseOptimized(True)
    info(useOptimized())


if __name__ == '__main__':
    basicConfig(stream=stdout, level=DEBUG)
    main()
