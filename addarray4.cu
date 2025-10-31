#include <iostream>
#include <iomanip>
#include <array>
#include <cuda_runtime.h>

__global__ void addarray(float* a, float* b, float* c, size_t dataSize){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < dataSize) {
		c[idx] = a[idx] + b[idx];
	}
}


int main(){
	constexpr size_t threadsPerBlock = 256;

	// Set up the data
	std::array<float, 5> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	std::array<float, 5> b = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
	std::array<float, 5> c{0};
	constexpr int dataSize = a.size();

	constexpr size_t blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;

	dim3 blockDim(threadsPerBlock);
	dim3 gridDim(blocksPerGrid);

	std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
	std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
	// Allocate the device memory
	float* cuda_a, *cuda_b, *cuda_c;
	cudaMalloc(&cuda_a, a.size() * sizeof(float));
	cudaMalloc(&cuda_b, b.size() * sizeof(float));
	cudaMalloc(&cuda_c, c.size() * sizeof(float));
	// Copy the data to the device
	cudaMemcpy(cuda_a, a.data(), a.size() * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);
	// Launch the kernel
	addarray<<<blockDim, blockDim>>>(cuda_a, cuda_b, cuda_c, dataSize);
	cudaDeviceSynchronize();
	// Copy the data back to the host
	cudaMemcpy(c.data(), cuda_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost);
	// Free the device memory
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	// Print the results

	int index = 0;
	std::cout << std::setw(10)
		  << std::left
		  << "Index"
		  << std::setw(10)
		  << std::left
		  << "Value"
		  << std::endl;
	for (auto& num : c) {
		std::cout << std::setw(10)
		  << std::left
		  << index
		  << std::setw(10)
		  << std::left
			  <<std::fixed
		  << std::setprecision(2)
		  << num
		  << std::endl;
		++index;
	}
	std::cout << std::endl;

	return 0;
}



