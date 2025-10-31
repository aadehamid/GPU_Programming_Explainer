#include <iostream>
#include <cuda_runtime.h>
#include <array>
#include <iomanip>

__global__ void scalarMult2DArray(float* a,float* result, int width, int height){
	// calculate the global thread index
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	
	// Flatten 2D index into 1D index to access array
	int idx = row * width + col;

	if (col < width && row < height){
		result[idx] = a[idx] * 2.f;
	}
}

int main(){
	constexpr size_t WIDTH = 7;
	constexpr size_t HEIGHT = 5;

	std::array<std::array<float, WIDTH>, HEIGHT> a {{
		{ 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f},  // Row 0
		{ 8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f},  // Row 1
		{15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f},  // Row 2
		{22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f},  // Row 3
		{29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f}   // Row 4
	}};
	std::array<float, WIDTH * HEIGHT> result;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
	);


	// allocate device memory
	float* cuda_a = nullptr;
	float* cuda_result = nullptr;
	cudaMalloc(&cuda_a, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc(&cuda_result, WIDTH * HEIGHT * sizeof(float));

	// Copy data from host to device
	cudaMemcpy(cuda_a, a.data(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	
	// Launch the kernel
	scalarMult2DArray<<<blocksPerGrid, threadsPerBlock>>>(cuda_a, cuda_result, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
	
	// Copy result from device to host
	cudaMemcpy(result.data(), cuda_result, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(cuda_a);
	cudaFree(cuda_result);

	// print result
	for (int i = 0; i < WIDTH * HEIGHT; i++){
		std::cout << std::setw(5) << result[i] << " ";
		if (i % WIDTH == WIDTH - 1){
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	return 0;


      
}
