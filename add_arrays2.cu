//Declare required CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// C++ specific headers
#include <iostream>

// Function declarations
__global__ void vectorized(int a[], int b[], int c[], size_t size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size){
    c[i] = a[i] + b[i];
    }
return;
}

int main(){
    size_t size = 4;
    int a[size]{1,2,3,4};
    int b[size]{5,6,7,8};
    int c[size]{0};

    // Set up pointers to hold memory on GPU
    int* cuda_a = nullptr;
    int* cuda_b = nullptr;
    int* cuda_c = nullptr;

    // Allocate the memory on GPU
    cudaMalloc(&cuda_a, sizeof(int) * size);
    cudaMaloc(&cuda_b, sizeof(int) * size);
    cudaMalloc(&cuda_c, sizeof(int) * size);

    // Move data from HOST to DEVICE
    cudaMemcpy(cuda_a, a, sizeof(a), cudaMemcpyHostToDevice)
    cudaMemcpy(cuda_b, sizeof(b), cudaMemcpyHostToDevice);


    // Call the function on GPU
    vectorized<<<1, sizeof(a)/sizeof(int)>>>(cuda_a, cuda_b, cuda_c, size);
    cudaDeviceSynchronize();

    // Move data from GPU to CPU
    cudaMemcpy(c, cuda_c, sizeof(c), cudaMemcpyDeviceToHost);

    // Print the result
    for (size_t i{0}; i < size; i++){

    std::cout<< c[i] << "  ";
    }

    // Free GPU Memory
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);


}