cmake_minimum_required(VERSION 3.18)
project(flash_attention CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(flash_attn flash_attn.cu)
set_target_properties(flash_attn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
