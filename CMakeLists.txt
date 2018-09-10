cmake_minimum_required (VERSION 3.10.2)
project (Open-GPGPU-ANN)

include_directories(../../arrayfire-3/include/af)
link_directories(../../arrayfire-3/lib)

include_directories(../arrayfire-ml/include)
link_directories(../arrayfire-ml/cmake-build-debug)


add_executable(test_arrayfire_ml test_arrayfire_ml.cpp)
add_executable(test_SYCL test_SYCL.cpp)