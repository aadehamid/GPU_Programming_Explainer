// dynamic_shared_vector_add.cu

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256
#define BLOCK_SIZE 64

// Kernel using dynamically allocated shared memory
__global__ void vectorAddDynamic(const float* A, const float* B, float* C) {
    // Declare dynamic shared memory (size set at kernel launch)
    extern __shared__ float shared[];

    // Split shared memory for A and B
    float* sA = shared;
    float* sB = &shared[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory to shared memory
    if (idx < N) {
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];
    }
    __syncthreads();

    // Perform addition in shared memory
    if (idx < N) {
        C[idx] = sA[threadIdx.x] + sB[threadIdx.x];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Each block needs 2 * BLOCK_SIZE floats in shared memory
    size_t sharedMemSize = 2 * BLOCK_SIZE * sizeof(float);

    vectorAddDynamic<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
