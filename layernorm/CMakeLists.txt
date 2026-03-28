cmake_minimum_required(VERSION 3.18)
project(layernorm_cuda CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 是 Ampere sm_86

add_executable(layernorm layernorm.cu)
set_target_properties(layernorm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
