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

__global__ void staticSharedExample(float* output) {
    __shared__ float tile[32];

    int tid = threadIdx.x;  // Fixed typo here
    tile[tid] = tid * 2;
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

    // Copy shared memory back to global memory
    // so the host can see it
    output[tid] = tile[tid];
}

int main() {
    constexpr int WIDTH = 32;
    dim3 threadsPerBlock(WIDTH, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

    float* d_out;
    float* h_out = new float[WIDTH];  // Allocate host memory for the output
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_out, WIDTH * sizeof(float)));

    // Launch kernel with error checking
    staticSharedExample<<<blocksPerGrid, threadsPerBlock>>>(d_out);
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Synchronize and check for errors
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results from host
    std::cout << "Results from device:" << std::endl;
    for (int i = 0; i < WIDTH; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }
    
    // Clean up
    delete[] h_out;
    CHECK_CUDA(cudaFree(d_out));
    
    return 0;
}
