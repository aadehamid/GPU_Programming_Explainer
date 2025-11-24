#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iomanip>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

constexpr int TILE_ROWS = 8;
constexpr int TILE_COLS = 16;
constexpr int WIDTH = 32;
constexpr int HEIGHT = 64;

__global__ void vectorSharedAdd(const float *a, const float *b, float *c,
                             const int width, const int height) {
    __shared__ float s_a[TILE_ROWS][TILE_COLS];
    __shared__ float s_b[TILE_ROWS][TILE_COLS];

	// When in the tile the local index in rows and column
	// is determined by threadIdx.y and threadIdx.x, respectively
	int localRow = threadIdx.y;
	int localCol = threadIdx.x;

	int matrixRow = blockIdx.y * TILE_ROWS + localRow;
	int matrixCol = blockIdx.x * TILE_COLS + localCol;

	if (matrixRow < height && matrixCol < width){
		// calculate the global tread index to store the result
		int globalThreadIndex = matrixRow * width + matrixCol;

		// Load the data into the shared memory
		s_a[localRow][localCol] = a[globalThreadIndex];
		s_b[localRow][localCol] = b[globalThreadIndex];
	} else {
		// If the thread is out of bounds, set the shared memory to 0
		s_a[localRow][localCol] = 0.0f;
		s_b[localRow][localCol] = 0.0f;
	}

	// Synchronize to make sure the data is loaded
	__syncthreads();

	// Perform the addition
	if (matrixRow < height && matrixCol < width){

		// calculate the global tread index to store the result
		int globalThreadIndex = matrixRow * width + matrixCol;
		c[globalThreadIndex] = s_a[localRow][localCol] + s_b[localRow][localCol];
		}


}

int main(){

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	// create the 2D array
	std::vector<std::vector<float>> a(HEIGHT, std::vector<float>(WIDTH));
	std::vector<std::vector<float>> b(HEIGHT, std::vector<float>(WIDTH));
	std::vector<std::vector<float>> c(HEIGHT, std::vector<float>(WIDTH));

	for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < WIDTH; j++){
			a[i][j] = dis(gen);
			b[i][j] = dis(gen);
		}
	}

	dim3 threadPerBlock(TILE_COLS, TILE_ROWS);
	dim3 blockPergrid((WIDTH + TILE_COLS -1)/TILE_COLS, (HEIGHT + TILE_ROWS -1)/TILE_ROWS);

    // Create contiguous memory for device
    std::vector<float> a_contig, b_contig, c_contig(HEIGHT * WIDTH);
    
    // Copy 2D vector to contiguous memory
    a_contig.reserve(HEIGHT * WIDTH);
    b_contig.reserve(HEIGHT * WIDTH);
    
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            a_contig.push_back(a[i][j]);
        }
    }
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            b_contig.push_back(b[i][j]);
        }
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, sizeof(float) * HEIGHT * WIDTH));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * HEIGHT * WIDTH));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * HEIGHT * WIDTH));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_a, a_contig.data(), sizeof(float) * HEIGHT * WIDTH, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_contig.data(), sizeof(float) * HEIGHT * WIDTH, cudaMemcpyHostToDevice));

	// Launch the kernel
	std::cout << "blockDim = (" << threadPerBlock.x << ", " << threadPerBlock.y << ")\n";
	std::cout << "blockPergrid = (" << blockPergrid.x << ", " << blockPergrid.y << ")\n";

	// launch the kernel
	vectorSharedAdd<<<blockPergrid, threadPerBlock>>>(d_a, d_b, d_c, WIDTH, HEIGHT);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

    // Copy data back to host
    CHECK_CUDA(cudaMemcpy(c_contig.data(), d_c, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyDeviceToHost));
    
    // Copy back to 2D vector
    for (int i = 0; i < HEIGHT; ++i) {
        std::copy_n(c_contig.begin() + i * WIDTH, WIDTH, c[i].begin());
    }

	// Verify results on CPU for the first few elements
	bool all_correct = true;
	for (int i = 0; i < std::min(4, HEIGHT); i++) {
	    for (int j = 0; j < std::min(8, WIDTH); j++) {
	        float expected = a[i][j] + b[i][j];
	        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << c[i][j];
	        
	        // Check if the result matches CPU computation
	        if (std::abs(c[i][j] - expected) > 1e-5) {
	            std::cout << "* ";  // Mark incorrect results
	            all_correct = false;
	        } else {
	            std::cout << "  ";
	        }
	    }
	    std::cout << "\n";
	}
	
	if (all_correct) {
	    std::cout << "All results are correct!\n";
	} else {
	    std::cout << "* Incorrect results detected!\n";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;

}

