// static_shared_vector_add.cu

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256  // Vector size
#define BLOCK_SIZE 64  // Threads per block

// Kernel using statically allocated shared memory
__global__ void vectorAddStatic(const float* A, const float* B, float* C) {
    // Statically allocated shared memory (size known at compile time)
    __shared__ float sA[BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory to shared memory
    if (idx < N) {
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];
    }
    __syncthreads(); // Ensure all data is loaded

    // Perform addition in shared memory
    if (idx < N) {
        C[idx] = sA[threadIdx.x] + sB[threadIdx.x];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAddStatic<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some results
    for (int i = 0; i < 5; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
