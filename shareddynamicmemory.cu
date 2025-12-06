#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>

#define EPS   1e-3f    // epsilon for floating point comparison

constexpr int WIDTH = 32;
constexpr int HEIGHT = 64;
constexpr int TILE_ROWS = 8;
constexpr int TILE_COLS = 16;

__global__ void vecttorSharedAdd(const float *a, const float *b, float *c, const int width, const int height){
	// declare the shared memory
	extern __shared__ float sharedMem[];

	// Divide the allocated buffer into two segments
	float *aShared = sharedMem;
	float *bShared = sharedMem + TILE_COLS * TILE_ROWS;

	// Determine the index
	int localTileIndex = threadIdx.y * TILE_COLS + threadIdx.x;

	int matrixRowIndex = blockIdx.y * TILE_ROWS + threadIdx.y;
	int matrixColIndex = blockIdx.x * TILE_COLS + threadIdx.x;

	if (matrixRowIndex < HEIGHT && matrixColIndex < WIDTH){
		// Load the data into shared memory
		aShared[localTileIndex] = a[matrixRowIndex * width + matrixColIndex];
		bShared[localTileIndex] = b[matrixRowIndex * width + matrixColIndex];
	}else {
		aShared[localTileIndex] = 0.0f;
		bShared[localTileIndex] = 0.0f;
	}
	__syncthreads();

	// Perform the addition
	
	if (matrixRowIndex < HEIGHT && matrixColIndex < WIDTH){
		c[matrixRowIndex * width + matrixColIndex] = aShared[localTileIndex] + bShared[localTileIndex];
	}
	
}

int main(){
	
	// set the kernel configuration
	dim3 threadsPerBlock(TILE_COLS, TILE_ROWS);
	dim3 blockPerGrid((WIDTH + TILE_COLS - 1) / TILE_COLS, (HEIGHT + TILE_ROWS - 1) / TILE_ROWS);


	// Create the array
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	std::vector<float> a(HEIGHT * WIDTH);
	std::vector<float> b(HEIGHT * WIDTH);
	std::vector<float> c(HEIGHT * WIDTH);

	// now fill in the data
	for (auto& val : a){
		val = dist(gen);
	}

	for (auto& val : b){
		val = dist(gen);
	}

	// Create memory for the data on the GPU
	int sharedBytes = 2 * TILE_COLS * TILE_ROWS * sizeof(float);

	// Create memory on the GPU
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, a.size() * sizeof(float));
	cudaMalloc(&d_b, b.size() * sizeof(float));
	cudaMalloc(&d_c, c.size() * sizeof(float));
	
	// Copy the data to the GPU
	cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);
	
	// Launch the kernel
	vecttorSharedAdd<<<blockPerGrid, threadsPerBlock, sharedBytes>>>(d_a, d_b, d_c, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
	// Copy the data back to the CPU
	cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost);

	// Print the results
	for (int row = 0; row < 8; ++row){
		for (int col = 0; col < 8; ++col){
			std::cout << std::fixed << std::setprecision(2) << c[row * WIDTH + col] << " ";
		}
		std::cout << std::endl;
	}

	// ---- 5) Verify -------------------------------------------------------
    bool correct = true;
    for (int i = 0; i < 8; ++i) {
        float expected = a[i] + b[i];
        float diff = fabsf(c[i] - expected);
        if (diff > EPS) {
            correct = false;
            printf("Mismatch at index %d: GPU=%f  CPU=%f  diff=%f\n",
                   i, c[i], expected, diff);
        }
    }

    if (correct)
        printf("Quick verification PASSED for %d elements.\n", 8);
    else
        printf("Quick verification FAILED.\n");
	

	//Free the memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	// return 0;
	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
	
}
