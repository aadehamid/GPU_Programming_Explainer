#include <iostream>
#include <cstdio>

// CUDA kernel function
__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main() {
    std::cout << "Launching CUDA kernel..." << std::endl;
    hello<<<2, 2>>>();
    cudaDeviceSynchronize();
    std::cout << "CUDA kernel finished." << std::endl;
    return 0;
}
