// cuda_error_check.h
#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <iostream>
#include <source_location>
#include <cstdlib>

/**
 * @brief Check CUDA API call errors with automatic source location tracking
 *
 * @param error The cudaError_t returned by a CUDA API call
 * @param location Source location (automatically captured at call site)
 *
 * @example
 * cudaCheckError(cudaMalloc(&d_data, size));
 * cudaCheckError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
 */
inline void cudaCheckError(
    cudaError_t error,
    const std::source_location& location = std::source_location::current()
) {
    if (error != cudaSuccess) {
        std::cerr << "====== CUDA API Error ======" << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "File: " << location.file_name() << std::endl;
        std::cerr << "Line: " << location.line() << std::endl;
        std::cerr << "Function: " << location.function_name() << std::endl;
        std::cerr << "============================" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Check CUDA kernel launch and execution errors
 *
 * This function performs two checks:
 * 1. cudaPeekAtLastError() - Checks for kernel launch configuration errors
 * 2. cudaDeviceSynchronize() - Waits for kernel completion and checks execution errors
 *
 * @param kernelName Name of the kernel (for error messages)
 *
 * @example
 * myKernel<<<gridDim, blockDim>>>(args);
 * checkKernelErrors("myKernel");
 */
inline void checkKernelError(const char* kernelName) {
    // Check for launch configuration errors
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        std::cerr << "====== Kernel Launch Error ======" << std::endl;
        std::cerr << "Kernel: " << kernelName << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "=================================" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Wait for kernel to complete and check for execution errors
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "====== Kernel Execution Error ======" << std::endl;
        std::cerr << "Kernel: " << kernelName << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "====================================" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#endif // CUDA_ERROR_CHECK_H