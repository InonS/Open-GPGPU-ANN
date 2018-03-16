from os import system

from pyopencl import get_cl_header_version, get_platforms
from pyopencl.tools import get_test_platforms_and_devices

print(system("clinfo"))

print(get_cl_header_version())

print(get_test_platforms_and_devices())

for platform in get_platforms():
    print(platform.get_info())
    for device in platform.get_devices():
        print(device.get_info())
