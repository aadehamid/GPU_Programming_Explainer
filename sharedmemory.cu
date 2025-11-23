#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

__global__ void staticSharedExample() {
    __shared__ float tile[32];

    int tid = threadIdx.x;  // Fixed typo here
    tile[tid] = tid;
    __syncthreads();

    if (tid == 0) {
        // Using printf in device code
        printf("Static Shared Memory Example\n");
        printf("Index | Value\n");
        printf("------|------\n");
        for (int i = 0; i < 32; i++) {
            printf("%5d | %5.0f\n", i, tile[i]);
        }
    }
}

int main() {
    constexpr int WIDTH = 32;
    dim3 threadsPerBlock(WIDTH, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

    // Launch kernel with error checking
    staticSharedExample<<<blocksPerGrid, threadsPerBlock>>>();
    
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Synchronize and check for errors
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
