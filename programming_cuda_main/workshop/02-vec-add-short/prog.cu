//
//  Created by Patricio Bulic, Davor Sluga, UL FRI on 6/6/2022.
//  Copyright © 2022 Patricio Bulic, Davor Sluga UL FRI. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

// CUDA kernel. Each thread takes care of one element of c
__global__ void vec_add(const float *a, const float *b, float *c, const int n)
{
    // Get global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char *argv[])
{
    // Size of vectors
    int n = 1024;
    size_t datasize = sizeof(float) * n;

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;

    // Device input vectors
    float *d_a;
    float *d_b;
    // Device output vector
    float *d_c;

    // Allocate memory for each vector on host
    h_a = (float *)malloc(datasize);
    h_b = (float *)malloc(datasize);
    h_c = (float *)malloc(datasize);

    // Allocate memory for each vector on GPU
    checkCudaErrors(cudaMalloc(&d_a, datasize));
    checkCudaErrors(cudaMalloc(&d_b, datasize));
    checkCudaErrors(cudaMalloc(&d_c, datasize));

    // Initialize vectors on host
    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    // Copy host vectors to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, datasize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, datasize, cudaMemcpyHostToDevice));

    // Define thread grid
    dim3 blockSize = 1024;
    dim3 gridSize = 1;

    // Execute the kernel
    vec_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    getLastCudaError("vec_add() execution failed\n");

    // Copy array back to host
    checkCudaErrors(cudaMemcpy(h_c, d_c, datasize, cudaMemcpyDeviceToHost));

    // Check the result
    float result = 0.0;
    for (int i = 0; i < n; i++)
        result += h_c[i];
    printf("Result = %.2f \n", result);

    // Release device memory
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
