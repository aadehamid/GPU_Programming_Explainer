//
// Created by Hamid Adesokan on 9/22/25.
//

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int nonVectorized(int arr1[], int arr2[], int arr3[], int size){
	for (int i = 0; i < size; i++){
		arr3[i] = arr1[i] + arr2[i];
	}
	return 0;
}

__global__ void vectorized(int arr1[], int arr2[], int arr3[], size_t size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size){
	arr3[i] = arr1[i] + arr2[i];
		}
	return;
}


int main(){
	int a[3] {1,2,3};
	int b[3] {4,5,6};
	int c[sizeof(a)/sizeof(int)]{0};

	// nonVectorized(a, b, c, sizeof(a)/sizeof(int));

	// for (size_t i{0}; i < (sizeof(a)/sizeof(int)); i++){
	// 	c[i] = a[i] + b[i];
	// }

	// for (size_t i{0}; i < (sizeof(a)/sizeof(int)); i++){
	// 	std::cout << c[i] << " ";
	// }
	
	int* cuda_arr1 = nullptr;
	int* cuda_arr2 = nullptr;
	int* cuda_arr3 = nullptr;

	// Allocate memory on GPU
	cudaMalloc(&cuda_arr1, sizeof(a));
	cudaMalloc(&cuda_arr2, sizeof(b));
	cudaMalloc(&cuda_arr3, sizeof(c));

	// Copy data from CPU to GPU
	cudaMemcpy(cuda_arr1, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_arr2, b, sizeof(b), cudaMemcpyHostToDevice);
	// cudaMemcpy(cuda_arr3, c, sizeof(c), cudaMemcpyHostToDevice);
	
	vectorized<<<1,sizeof(a)/sizeof(int)>>>(cuda_arr1, cuda_arr2, cuda_arr3, sizeof(a)/sizeof(int));
	cudaDeviceSynchronize();
	// Copy data from GPU to CPU
	cudaMemcpy(c, cuda_arr3, sizeof(c), cudaMemcpyDeviceToHost);

	for (size_t i{0}; i < (sizeof(c)/sizeof(int)); i++){
		std::cout << c[i] << " ";
	}

	// Free memory on GPU
	cudaFree(cuda_arr1);
	cudaFree(cuda_arr2);
	cudaFree(cuda_arr3);
}
