//
//  Created by Patricio Bulic, Davor Sluga, UL FRI on 6/6/2022.
//  Copyright © 2022 Patricio Bulic, Davor Sluga UL FRI. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#define BLOCK_SIZE 64

// CUDA kernel. Each thread takes care of one element of c, each thread block preforms reduction
__global__ void dotProduct(const float *a, const float *b, float *c, int n)
{
    // Get global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float buf[BLOCK_SIZE];

    // Initialize shared buffer
    buf[threadIdx.x] = 0;

    // Make sure we do not go out of bounds
    while (tid < n)
    {
        buf[threadIdx.x] += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x;
    }

    __syncthreads();

    // Reduction:
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            // perform the addition:
            buf[threadIdx.x] += buf[threadIdx.x + i];
        }
        i = i / 2;
        // wait at the barrier:
        __syncthreads();
    }

    // only one thread in block writes the result:
    if (threadIdx.x == 0)
    {
        c[blockIdx.x] = buf[0];
    }
}

int main(int argc, char *argv[])
{
    // Size of vectors
    int n = 1024;
    dim3 blockSize = BLOCK_SIZE;
    dim3 gridSize = n / blockSize.x;

    size_t datasize = sizeof(float) * n;
    size_t partial_results_size = sizeof(float) * gridSize.x;

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
    h_c = (float *)malloc(partial_results_size);

    // Allocate memory for each vector on GPU
    checkCudaErrors(cudaMalloc(&d_a, datasize));
    checkCudaErrors(cudaMalloc(&d_b, datasize));
    checkCudaErrors(cudaMalloc(&d_c, partial_results_size));

    // Initialize vectors on host
    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    // Copy host vectors to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, datasize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, datasize, cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event on the stream
    cudaEventRecord(start);
    
    // Execute the kernel
    dotProduct<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // record stop event on the stream
    cudaEventRecord(stop);
    getLastCudaError("dotProduct() execution failed\n");

    // wait until the stop event completes
    cudaEventSynchronize(stop);    
    
    // Calculate the elapsed time between two events
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);
    
    // Copy array back to host
    checkCudaErrors(cudaMemcpy(h_c, d_c, partial_results_size, cudaMemcpyDeviceToHost));

    // Sum up the products
    double result = 0.0;
    for (int i = 0; i < gridSize.x; i++)
        result += h_c[i];
    printf("Result = %.2f \n", result);

    // Release device memory
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    
    // Clean up the two events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
