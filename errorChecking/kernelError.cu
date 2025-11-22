#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda_error_check.h>

// inline void checkKernelError(const char* kernelName){
// 	cudaError_t error = cudaPeekAtLastError();
//
// 	if(error != cudaSuccess){
// 		std::cerr << "=== Kernel Launch Error Details ===" << std::endl;
// 		std::cerr << "Kernel Name: " << kernelName << std::endl;
// 		std::cerr <<"Error: " << cudaGetErrorString(error) << std::endl;
//
// 		std::exit(EXIT_FAILURE);
// 		}
// 	error = cudaDeviceSynchronize();
//
// 	if(error != cudaSuccess){
// 		std::cerr << "=== Kernel Execution Error Details ===" << std::endl;
// 		std::cerr << "Kernel Name: " << kernelName << std::endl;
// 		std::cerr <<"Error: " << cudaGetErrorString(error) << std::endl;
//
// 		std::exit(EXIT_FAILURE);
// 		}
// }

__global__ void simpleKernel(float* data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[idx] = idx * 2.0f;
}

__global__ void outOfBoundsKernel(float* data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Intentionally access memory way beyond the allocated array
	// This will cause an illegal memory access
	data[idx * 1000] = idx;  // If we have 1280 threads but only 1024 elements...
}


// int main() {
// 	float* d_data;
// 	cudaMalloc(&d_data, 1024 * sizeof(float));
//
// 	std::cout << "Test 1: Invalid block size (too many threads per block)" << std::endl;
//
// 	// Most GPUs have a max of 1024 threads per block
// 	// Let's try to launch with way more than that
// 	simpleKernel<<<10, 2048>>>(d_data);  // 2048 threads - will fail!
// checkKernelError("simpleKernel");
//
// 	// This line won't be reached
// 	std::cout << "This won't print" << std::endl;
//
// 	cudaFree(d_data);
// 	return 0;
// }

int main() {
	float* d_data;
	int size = 1024;
	cudaMalloc(&d_data, size * sizeof(float));

	std::cout << "Test 2: Out of bounds memory access" << std::endl;

	// Launch with 10 blocks * 128 threads = 1280 threads
	// But we only allocated space for 1024 floats
	// And we're accessing at idx * 1000, so we'll definitely go out of bounds
	outOfBoundsKernel<<<10, 128>>>(d_data, size);
	checkKernelError("outOfBoundsKernel");

	// This won't be reached
	std::cout << "This won't print" << std::endl;

	cudaFree(d_data);
	return 0;
}

