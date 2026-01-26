cmake_minimum_required(VERSION 3.18)

# Set CUDA toolkit root and compiler
set(CUDAToolkit_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(GPU_Programming_Explainer CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(main main.cu)
add_executable(shareddynamicmemory shareddynamicmemory.cu)

set_target_properties(main shareddynamicmemory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
