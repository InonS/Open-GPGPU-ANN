from enum import IntEnum
from time import perf_counter

from sys import stdout
from logging import basicConfig, debug, DEBUG, exception

from numpy import empty, float32, int32
from pyopencl import Buffer, CommandQueue, Context, Program, enqueue_copy, get_platforms, mem_flags
from scipy.misc import imread, imsave, imshow


class CL_DEVICE_TYPE(IntEnum):
    """
    see class device_type
    """
    DEFAULT = 1 << 0
    CPU = 1 << 1
    GPU = 1 << 2
    ACCELERATOR = 1 << 3
    ALL = (1 << 32) - 1


def get_devices():
    """
    Get devices for both CPU and GPU platforms
    """
    platforms = get_platforms()
    cpu_devices = platforms[0].get_devices()
    try:
        gpu_devices = platforms[1].get_devices()
    except IndexError:
        gpu_devices = None

    devices = gpu_devices if gpu_devices and len(gpu_devices) > 0 else cpu_devices
    debug(devices)
    return devices


def read_kernel_source(cl_source_file):
    with open(cl_source_file) as file:
        return '\n'.join(file.readlines())


def copy_global_read_only_buffer_from_host_pointer(context, hostbuf):
    return Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)


def allocate_variables(context, image):
    """
    Allocate memory for variables on the device
    """
    input_g = copy_global_read_only_buffer_from_host_pointer(context, image)

    width, height = image.shape
    width_g = copy_global_read_only_buffer_from_host_pointer(context, int32(width))
    height_g = copy_global_read_only_buffer_from_host_pointer(context, int32(height))

    result_g = Buffer(context, mem_flags.WRITE_ONLY, image.nbytes)

    return input_g, result_g, width_g, height_g


def copy_from_buffer(queue, buffer, shape, dtype):
    result = empty(shape, dtype)
    enqueue_copy(queue, result, buffer)
    return result


def main():
    devices = get_devices()
    try:
        debug(CL_DEVICE_TYPE(devices[0].type))
    except IndexError as ie:
        exception(ie)
    context = Context(devices)
    queue = CommandQueue(context)  # Create queue for each kernel execution

    source = read_kernel_source("median_filter.cl")
    program = Program(context, source).build()  # Kernel function instantiation

    image = imread('../data/noisyImage.jpg', flatten=True).astype(float32)  # Read in image
    imshow(image)

    start_usec = perf_counter()
    args = allocate_variables(context, image)

    program.medianFilter(queue, image.shape, None, *args)  # Call Kernel.
    # Automatically takes care of block/grid distribution. Note explicit naming of kernel to execute.

    result = copy_from_buffer(queue, args[1], image.shape, image.dtype)  # Copy the result back from buffer
    debug("%g milliseconds" % (1e3 * (perf_counter() - start_usec)))

    imshow(result)
    imsave('../data/medianFilter-OpenCL.jpg', result)  # Show the blurred image


if __name__ == '__main__':
    basicConfig(level=DEBUG, stream=stdout)
    main()
