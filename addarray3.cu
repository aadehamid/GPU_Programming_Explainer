#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void addarray(float* a, float* b, float* c, int size){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < size){
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    std::array<float, 5> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::array<float, 5> b = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::array<float, 5> c{0};

    // Create device pointers for raw float arrays
    float* cuda_a = nullptr;
    float* cuda_b = nullptr;
    float* cuda_c = nullptr;

    // Allocate memory on the device (correct size calculation)
    cudaMalloc(&cuda_a, a.size() * sizeof(float));
    cudaMalloc(&cuda_b, b.size() * sizeof(float));
    cudaMalloc(&cuda_c, c.size() * sizeof(float));

    // Move data to the allocated memory
    cudaMemcpy(cuda_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;

    // Pass pointers directly (no dereferencing!)
    addarray<<<blocksPerGrid, threadsPerBlock>>>(cuda_a, cuda_b, cuda_c, a.size());
    cudaDeviceSynchronize();

    // Move data from the allocated memory back to host
    cudaMemcpy(c.data(), cuda_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::setw(10) << std::left << "Index"
              << std::setw(10) << std::left << "Value" << std::endl;

    int index = 0;
    for (const auto& num : c) {
        std::cout << std::setw(10) << std::left << index++
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << std::left << num << std::endl;
    }

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    return 0;
}