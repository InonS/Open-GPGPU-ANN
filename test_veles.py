# wget -O - https://velesnet.ml/ubuntu-install.sh | bash -

from os.path import join as path_join
from sys import path

VELES_PATH = path_join("home", "inon", "src", "veles", "build", "lib", "veles")
path.append(VELES_PATH)

from numpy import float32, zeros
from veles.accelerated_units import IOpenCLUnit
from veles.backends import OpenCLDevice
from veles.memory import Array

# from veles.numpy_ext import roundup

from veles.ocl_blas import OCLBLAS

kibi = 1 << 10

# accelerated_workflow = AcceleratedWorkflow(workflow)
# accelerated_unit = AcceleratedUnit(workflow)

# auto_device = AutoDevice()
ocl_device = OpenCLDevice()
# accelerated_unit.device = ocl_device

# kernel_builder = Builder(workflow)
kernel_builder = OCLBLAS(ocl_device)


class MyOCL(IOpenCLUnit):
    def __init__(self):
        self.a = Array(zeros([kibi >> 1, kibi], dtype=float32))
        self.b = Array()
        self.b.mem = zeros([kibi, kibi], dtype=float32)

    def initialize(self, device, **kwargs):
        self.a.initialize(self)
        self.b.initialize(self)

        def ocl_init():
            self.krn_.set_arg(0, self.a.devmem)
            self.krn_.set_arg(1, self.b.devmem)

        ocl_init()

    def __call__(self, *args, **kwargs):
        self.a.unmap()
        self.b.unmap()
        self.execute_kernel(global_size, local_size, self.krn_)

        a = self.a.ocl_map_read()
