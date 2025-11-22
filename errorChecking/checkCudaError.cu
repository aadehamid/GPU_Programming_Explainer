#include <stdio.h>
#include <source_location>
#include <cstdlib>
#include <iostream>

inline void cuda_check_impl(cudaError_t error,
const std::source_location& location = std::source_location::current()){
    if (error != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) 
                    << " at "
                    << location.file_name()
                    << ":" 
                    << location.line() 
                    << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void hello() {
    printf("Hello, world!\n");
}

void test_macro_version() {
    std::cout << "=== Testing Macro Version ===" << std::endl;
    
    float* d_data;
    
    // This will succeed
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
    std::cout << "Successfully allocated memory" << std::endl;
    
    // Intentional error #1: Try to allocate an impossibly large amount of memory
    // This will trigger cudaErrorMemoryAllocation
    float* d_huge;
    CUDA_CHECK(cudaMalloc(&d_huge, (size_t)1e15));  // 1 petabyte - will fail!
    
    // This line won't be reached because the program exits on error
    std::cout << "This won't print" << std::endl;
}

void test_modern_version() {
    std::cout << "=== Testing Modern Version ===" << std::endl;
    
    float* d_data;
    
    // This will succeed
    cuda_check_impl(cudaMalloc(&d_data, 1024 * sizeof(float)));
    std::cout << "Successfully allocated memory" << std::endl;
    
    // Intentional error #1: Try to allocate too much memory
    float* d_huge;
    cuda_check_impl(cudaMalloc(&d_huge, (size_t)1e15));  // Will fail!
    
    // This line won't be reached
    std::cout << "This won't print" << std::endl;
}

int main() {
    hello<<<1, 1>>>();
    // CUDA_CHECK(cudaDeviceSynchronize(1));
    // test_macro_version();
    test_modern_version();
    cudaDeviceSynchronize();
    return 0;
}