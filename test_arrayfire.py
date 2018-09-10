# utf-8

"""
TODO http://arrayfire.org/docs/machine_learning_2deep_belief_net_8cpp-example.htm
TODO https://github.com/arrayfire/arrayfire-ml/tree/master/examples
TODO https://github.com/arrayfire/forge
"""
from logging import DEBUG, basicConfig, info
from sys import stdout
from timeit import default_number, timeit

import arrayfire as af
from numpy import uint16, uint32, uint8


# http://arrayfire.org/arrayfire-python
def calc_pi_device(samples):
    # Simple, array based API
    # Generate uniformly distributed random numers
    x = af.randu(samples)
    y = af.randu(samples)
    # Supports Just In Time Compilation
    # The following line generates a single kernel
    within_unit_circle = (x * x + y * y) < 1
    # Intuitive function names
    return 4 * af.count(within_unit_circle) / samples


def benchmark_current_backend(n_samples: uint32 = uint8(1e3), n_repetitions: uint16 = uint16(1 << 12)):
    stmt: str = "calc_pi_device(n_samples)"
    globals_ = {"calc_pi_device": calc_pi_device, "n_samples": n_samples}
    total_seconds = timeit(stmt, number=min(n_repetitions, default_number), globals=globals_)
    info("{}: {} milliseconds".format(af.get_backend(), 1000 * total_seconds / n_repetitions))


def main():
    for backend in af.get_available_backends():
        af.set_backend(backend, unsafe=True)
        benchmark_current_backend()


if __name__ == '__main__':
    basicConfig(stream=stdout, level=DEBUG)
    main()
