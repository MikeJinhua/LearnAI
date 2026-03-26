cmake_minimum_required(VERSION 3.18)
project(softmax_cuda CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 是 Ampere sm_86

add_executable(softmax softmax.cu)
set_target_properties(softmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
) 