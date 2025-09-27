// This program demonstrates vector addition using CUDA on the GPU.
// It adds two arrays element-wise in parallel on the GPU and retrieves the result on the CPU.
// Each step is commented for educational clarity.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// C++ specific headers
#include <iostream>

// CUDA kernel function to add two arrays element-wise
// Each thread computes one element of the result array
__global__ void vectorized(int a[], int b[], int c[], size_t size){
    // Calculate the global index for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Make sure we don't go out of bounds
    if (i < size){
        // Perform the addition for element i
        c[i] = a[i] + b[i];
    }
}

int main(){
    // Number of elements in our arrays
    size_t size = 4;

    // Host arrays (CPU memory)
    int a[size]{1,2,3,4}; // First input array
    int b[size]{5,6,7,8}; // Second input array
    int c[size]{0};       // Output array to hold results

    // Device pointers (GPU memory)
    int* cuda_a = nullptr;
    int* cuda_b = nullptr;
    int* cuda_c = nullptr;

    // Allocate memory on the GPU for each array
    cudaMalloc(&cuda_a, sizeof(int) * size);
    cudaMalloc(&cuda_b, sizeof(int) * size);
    cudaMalloc(&cuda_c, sizeof(int) * size);

    // Copy input data from CPU (host) to GPU (device)
    cudaMemcpy(cuda_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel with one block of 'size' threads
    // Each thread will compute one element of the output array
    vectorized<<<1, size>>>(cuda_a, cuda_b, cuda_c, size);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // Copy the result from GPU back to CPU
    cudaMemcpy(c, cuda_c, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Print the result array
    std::cout << "Result of vector addition: ";
    for (size_t i{0}; i < size; i++){
        std::cout << c[i] << "  ";
    }
    std::cout << std::endl;

    // Free the GPU memory
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    return 0;

}