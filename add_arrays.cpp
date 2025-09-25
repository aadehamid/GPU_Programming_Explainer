//
// Created by Hamid Adesokan on 9/22/25.
//

#include <iostream>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

int main(){
	int a[3] {1,2,3};
	int b[3] {4,5,6};
	int c[sizeof(a)/sizeof(int)]{0};

	for (size_t i{0}; i < (sizeof(a)/sizeof(int)); i++){
		c[i] = a[i] + b[i];
	}

	for (size_t i{0}; i < (sizeof(a)/sizeof(int)); i++){
		std::cout << c[i] << " ";
	}
}
