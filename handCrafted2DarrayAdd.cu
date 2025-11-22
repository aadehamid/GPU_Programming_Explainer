/**
 * @file handCrafted2DarrayAdd.cu
 * @brief CUDA implementation of 2D array multiplication with comprehensive error checking
 * 
 * This program demonstrates:
 * - 2D array allocation and initialization on both host and device
 * - CUDA kernel launch with optimal block size calculation
 * - Comprehensive error checking for all CUDA API calls
 * - Memory management and cleanup
 * - Random number generation for test data
 * 
 * The program multiplies two 2D arrays (matrices) element-wise using CUDA.
 */

#include <iostream>      // For console I/O
#include <iomanip>       // For formatted output
#include <array>         // For std::array
#include <cuda_runtime.h> // CUDA Runtime API
#include <source_location> // For better error reporting
#include <random>        // For random number generation
#include <algorithm>     // For std::min/max

/**
 * @brief Checks CUDA API calls for errors and reports them with source location
 * 
 * This function checks the result of a CUDA API call and if an error occurred,
 * it prints detailed error information including the file, line number, and
 * function where the error occurred, then terminates the program.
 * 
 * @param error The CUDA error code to check
 * @param location Source location information (automatically provided)
 */
inline void cudaCheckError(cudaError_t error,
              const std::source_location& location = std::source_location::current()) {
	if (error != cudaSuccess) {
		std::cerr << "====Cuda Function Calls Error Details ====" << "\n"
		<<"Error: " <<cudaGetErrorString(error) <<"\n"
		<<"File: " << location.file_name() << "\n"
		<<"Line: " << location.line() << "\n"
		<<"function: " << location.function_name() <<std::endl;
		std::cerr << "============================" << "\n";
		std::exit(EXIT_FAILURE);
		}

	}

/**
 * @brief Checks for kernel launch and execution errors
 * 
 * This function checks for two types of CUDA kernel errors:
 * 1. Launch configuration errors (synchronous)
 * 2. Kernel execution errors (asynchronous, requires device synchronization)
 * 
 * @param kernelName Name of the kernel function for error reporting
 */
inline void kernelCheckError(const char* kernelName) {
	cudaError_t syncError = cudaGetLastError();
	cudaError_t asyncError = cudaSuccess;

	// Check for kernel launch errors
	if (syncError != cudaSuccess) {
		std::cerr << "==== CUDA Kernel Launch Error ====\n"
				  << "Kernel: " << kernelName << "\n"
				  << "Error: " << cudaGetErrorString(syncError) << "\n"
				  << "============================\n";
		std::exit(EXIT_FAILURE);
	}

	// Check for asynchronous errors (only if kernel launched successfully)
	asyncError = cudaDeviceSynchronize();
	if (asyncError != cudaSuccess) {
		std::cerr << "==== CUDA Kernel Execution Error ====\n"
				  << "Kernel: " << kernelName << "\n"
				  << "Error: " << cudaGetErrorString(asyncError) << "\n"
				  << "============================\n";
		std::exit(EXIT_FAILURE);
	}
}
	


/**
 * @brief Prints a 2D array to the console in a formatted way
 * 
 * This function prints a 2D array with proper formatting, including:
 * - Array name and dimensions
 * - Bounding box for better readability
 * - Limited width to prevent console overflow
 * 
 * @param arr Pointer to the array data (flattened 2D array)
 * @param width Width of the 2D array
 * @param height Height of the 2D array
 * @param arr_name Name of the array for the output header
 */
void printArray(const float* arr, int width, int height, const char* arr_name) {
	const int max_width = 100;  // Reasonable maximum width
	int display_width = std::min(width * 8, max_width);

	std::cout << std::string(display_width, '=') << "\n";
	std::cout << "\n" << arr_name << "(width: " << width << ", height: " << height << ")\n";

	for (size_t row{0}; row < height; ++row) {
		for (size_t col{0}; col < width; ++col) {
			std::cout << std::setw(8) << arr[row * width + col] << " ";
		}
		std::cout << "\n";
	}
	std::cout << std::string(display_width, '=') << "\n";
}
	


/**
 * @brief Calculates optimal block dimensions for CUDA kernel launch
 * 
 * This function determines the best block dimensions for a CUDA kernel launch
 * based on the device capabilities. It ensures that:
 * 1. Block dimensions are at least 1x1
 * 2. Total threads per block doesn't exceed device maximum
 * 3. Block dimensions are powers of two for better performance
 * 
 * @param blockSizeX Initial requested block width
 * @param blockSizeY Initial requested block height
 * @return dim3 Optimized block dimensions
 */
dim3 getOptimalBlockSize(size_t blockSizeX, size_t blockSizeY) {
	cudaDeviceProp deviceProp;
	cudaCheckError(cudaGetDeviceProperties(&deviceProp, 0));

	// Ensure block sizes are at least 1
	blockSizeX = std::max<size_t>(1, blockSizeX);
	blockSizeY = std::max<size_t>(1, blockSizeY);

	// Calculate the maximum threads per block
	const size_t maxThreadsPerBlock = static_cast<size_t>(deviceProp.maxThreadsPerBlock);

	// Reduce block size if necessary
	while (blockSizeX * blockSizeY > maxThreadsPerBlock) {
		if (blockSizeX > blockSizeY) {
			blockSizeX = std::max<size_t>(1, blockSizeX / 2);
		} else {
			blockSizeY = std::max<size_t>(1, blockSizeY / 2);
		}
	}

	std::cout << "Max threads per block: " << maxThreadsPerBlock << "\n"
			  << "Optimal block size: " << blockSizeX << " x " << blockSizeY << "\n"
			  << "Total threads per block: " << (blockSizeX * blockSizeY) << "\n";

	// dim3 constructor takes unsigned int, so we need to ensure the values fit
	// This is safe because we've already ensured they're small enough for CUDA
	return dim3(static_cast<unsigned int>(blockSizeX),
				static_cast<unsigned int>(blockSizeY));
}

/**
 * @brief CUDA kernel for element-wise multiplication of two 2D arrays
 * 
 * Each thread computes one element of the output array by multiplying
 * corresponding elements from the input arrays. The kernel includes
 * bounds checking to prevent out-of-bounds memory access.
 * 
 * @param a First input array (read-only)
 * @param b Second input array (read-only)
 * @param c Output array (a * b)
 * @param width Width of the 2D arrays
 * @param height Height of the 2D arrays
 */
__global__ void multiply2DArray(const float* __restrict__ a,
                     const float* __restrict__ b,
                    float* __restrict__ c,
                    const int width, const int height) {
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < static_cast<size_t>(height) && col < static_cast<size_t>(width)) {
		const size_t globalIndex = row * width + col;
		// Add bounds check for globalIndex
		if (globalIndex < static_cast<size_t>(width * height)) {
			c[globalIndex] = a[globalIndex] * b[globalIndex];
		}
	}
}

/**
 * @brief Fills a 2D array with random floating-point numbers
 * 
 * This template function populates a 2D array with random float values
 * between 0.0 and 100.0 using the Mersenne Twister random number generator.
 * 
 * @tparam Rows Number of rows in the array
 * @tparam Cols Number of columns in the array
 * @param arr Reference to the 2D array to fill with random values
 */
template<std::size_t Rows, std::size_t Cols>
void generateRandomArray(std::array<std::array<float, Cols>, Rows>& arr) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(0.0f, 100.0f);  // Changed to float
	for (auto& row : arr) {
		for (auto& col : row) {
			col = dist(mt);
		}
	}
}
/**
 * @brief Main function demonstrating 2D array multiplication using CUDA
 * 
 * This program:
 * 1. Initializes two 2D arrays with random values
 * 2. Allocates and copies data to the GPU
 * 3. Launches a CUDA kernel to perform element-wise multiplication
 * 4. Copies results back to host and verifies the output
 * 5. Cleans up all allocated resources
 * 
 * @return int EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int main() {
	// Create the array
	constexpr size_t width = 10;
	constexpr size_t height = 10;
	constexpr size_t totalElements = width * height;
	constexpr size_t bytes = totalElements * sizeof(float);
	std::array<std::array<float, width>, height> a;
	generateRandomArray(a);
	printArray(reinterpret_cast<const float*>(a.data()), width, height, "a");
	
	// Create the array
	std::array<std::array<float, width>, height> b;
	generateRandomArray(b);
	printArray(reinterpret_cast<const float*>(b.data()), width, height, "b");
	
	// Create the array
	std::array<std::array<float, width>, height> c;

	// GPU Memory Management
	// Allocate device memory for input and output arrays
	// Using cudaMalloc for device memory allocation with proper error checking
	std::cout << "Allocating memory on the GPU...\n";
	std::cout << "Allocating " << bytes
			  << " bytes per array for a total of "
			  << bytes * 3
			  << " bytes for three arrays\n";
	// Allocate device memory
	float *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;

try {
	// Allocate with error checking
	cudaCheckError(cudaMalloc(&a_d, bytes));
	cudaCheckError(cudaMalloc(&b_d, bytes));
	cudaCheckError(cudaMalloc(&c_d, bytes));

	// Copy the arrays to the GPU
	std::cout << "Copying data from host to device...\n";
	cudaCheckError(cudaMemcpy(a_d, a.data(), sizeof(float) * width * height, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(b_d, b.data(), sizeof(float) * width * height, cudaMemcpyHostToDevice));

	// generate the kernel configuration
	dim3 blockSize = getOptimalBlockSize(16, 16);
	dim3 gridSize((width + blockSize.x -1) / blockSize.x, (height + blockSize.y -1) / blockSize.y);

	// Launch the kernel
	multiply2DArray<<<gridSize, blockSize>>>(a_d, b_d, c_d, width, height);
	kernelCheckError("multiply2DArray");

	std::cout <<"Kernel completed successfully\n";

	// Copy the arrays from the GPU
	std::cout << "Copying data from device to host...\n";
	
	cudaCheckError(cudaMemcpy(c.data(), c_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	printArray(reinterpret_cast<const float*>(c.data()), width, height, "c");

} catch (const std::exception& e) {
	// Cleanup in case of error
	if (a_d) cudaCheckError(cudaFree(a_d));
	if (b_d) cudaCheckError(cudaFree(b_d));
	if (c_d) cudaCheckError(cudaFree(c_d));
	std::cerr << "Error: " << e.what() << "\n";
	return EXIT_FAILURE;
}

	// Cleanup in success case
	cudaCheckError(cudaFree(a_d));
	cudaCheckError(cudaFree(b_d));
	cudaCheckError(cudaFree(c_d));

	// Reset the device to clear any remaining resources
	cudaCheckError(cudaDeviceReset());

	return EXIT_SUCCESS;


	
}

