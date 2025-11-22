#include <iostream>
#include <source_location>

inline void cudaCheckError(cudaError_t error,
const std::source_location& location = std::source_location::current() ){
if (error != cudaSuccess){
std::cerr << "=== CUDA Error Details===\n"
<< "Error: " << cudaGetErrorString(error) << "\n"
<< "File: " << location.file_name() << "\n"
<< "Function: " << location.function_name() << "\n"
<< "Line: " << location.line() << "\n"
<< std::endl;
std::exit(EXIT_FAILURE);
}
}


inline void cudaCheckInvalidDevice(cudaError_t error,
const std::source_location& location = std::source_location::current() ){
if (error == cudaErrorInvalidDevice){
std::cerr << "=== CUDA Error Details===\n"
<< "Error: " << cudaGetErrorString(error) << "\n"
<< "File: " << location.file_name() << "\n"
<< "Line: " << location.line() << "\n"
<< "Function: " << location.function_name() << "\n"
<< std::endl;
std::exit(EXIT_FAILURE);
}
}



void test_invalid_device(){
std::cout << "=== Testing Invalid Device===\n";
cudaCheckError(cudaSetDevice(1000));

std::cout << "This line won't be reached\n";
}

void test_invalid_copy() {
    std::cout << "=== Testing Invalid Memory Copy ===" << std::endl;

    float* d_data;
    cudaCheckError(cudaMalloc(&d_data, 1024 * sizeof(float)));

    // Intentional error #3: Copy from NULL pointer
    cudaCheckError(cudaMemcpy(d_data, nullptr, 1024 * sizeof(float),
                               cudaMemcpyHostToDevice));  // Will fail!

    std::cout << "This won't print" << std::endl;
}

int main(){
// test_invalid_device();
test_invalid_copy();
return 0;
}
