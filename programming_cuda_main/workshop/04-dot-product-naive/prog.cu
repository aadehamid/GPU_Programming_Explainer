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
__global__ void dotProduct(const float *a, const float *b, float *c, const int n)
{
    // TODO: Write the kernel body
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

    dim3 blockSize = 64;
    dim3 gridSize = n / blockSize.x;

    // Execute the kernel
    dotProduct<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    getLastCudaError("dotProduct() execution failed\n");

    // Copy array back to host
    checkCudaErrors(cudaMemcpy(h_c, d_c, datasize, cudaMemcpyDeviceToHost));

    // Sum up the products
    double result = 0.0;
    // TODO: sum up the products to final result
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
