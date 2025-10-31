#include <iostream>
#include <cuda_runtime.h>
#include <array>
#include <iomanip>
#include <cstdlib>

// ============================================================================
// ERROR CHECKING UTILITIES
// ============================================================================

// This macro wraps CUDA API calls and checks for errors. If an error occurs,
// it prints the error message, the file name, and line number, then exits.
// This is crucial for debugging because CUDA errors are often silent otherwise.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// This function checks for errors after kernel launches. Kernel launches
// themselves don't return error codes, so we need to explicitly check.
// We use cudaPeekAtLastError() to check if the launch configuration was valid,
// and cudaDeviceSynchronize() to catch any errors that occur during execution.
inline void checkKernelErrors(const char* kernelName) {
    // Check for launch configuration errors
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel Launch Error (" << kernelName << "): "
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Wait for kernel to complete and check for execution errors
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Kernel Execution Error (" << kernelName << "): "
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// KERNEL IMPLEMENTATION
// ============================================================================

// The kernel now uses const for read-only parameters, which documents our
// intent and allows the compiler to potentially optimize better. We also
// mark width and height as const since they don't change during execution.
__global__ void scalarMult2DArray(const float* a, const float* b, float* result,
                                   const int width, const int height) {
    // Calculate the global thread index in both dimensions.
    // Each thread figures out which column and row it's responsible for
    // by combining its position within the block (threadIdx) with which
    // block it's in (blockIdx) and how large each block is (blockDim).
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    int row = threadIdx.y + (blockIdx.y * blockDim.y);

    // Guard condition: only proceed if this thread's position is within
    // the actual data bounds. This is critical because we launch more threads
    // than we have data elements (for example, 256 threads for 35 elements).
    // Without this check, threads would access invalid memory locations.
    if (col < width && row < height) {
        // Convert the 2D coordinates (row, col) into a 1D array index.
        // We use row-major order: all elements of row 0, then row 1, etc.
        // For a position at (row, col), we skip 'row' complete rows (each
        // of size 'width'), then move 'col' positions into the current row.
        int idx = row * width + col;

        // Perform the element-wise multiplication. Each thread handles
        // exactly one element of the output array.
        result[idx] = a[idx] * b[idx];
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// This function prints a 2D array in a readable format. It helps us verify
// that our computation produced the expected results.
void print2DArray(const float* arr, int width, int height, const char* name) {
    std::cout << "\n" << name << " (" << height << "x" << width << "):\n";
    std::cout << std::string(width * 8, '-') << "\n";

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                      << arr[idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::string(width * 8, '-') << "\n";
}

// This function helps us choose optimal thread block dimensions based on
// the GPU's capabilities. Different GPUs have different optimal configurations.
dim3 getOptimalBlockSize(int width, int height) {
    // Query device properties to understand the GPU's limitations
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Most modern GPUs support up to 1024 threads per block.
    // For 2D problems, common configurations are:
    // - 32x32 = 1024 threads (maximum, good for large problems)
    // - 16x16 = 256 threads (good general-purpose choice)
    // - 8x8 = 64 threads (better for kernels with high register usage)

    // For this simple kernel, we'll use 16x16 as a balanced choice.
    // In production code, you'd benchmark different sizes for your specific use case.
    int blockX = 16;
    int blockY = 16;

    // Make sure we don't exceed the device's maximum threads per block
    while (blockX * blockY > prop.maxThreadsPerBlock) {
        blockX /= 2;
        blockY /= 2;
    }

    std::cout << "Using thread block size: " << blockX << "x" << blockY
              << " = " << blockX * blockY << " threads per block\n";

    return dim3(blockX, blockY);
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main() {
    // Define array dimensions as compile-time constants.
    // Using constexpr allows the compiler to optimize better and makes
    // the code more maintainable since these values are used in multiple places.
    constexpr size_t WIDTH = 7;
    constexpr size_t HEIGHT = 5;
    constexpr size_t TOTAL_ELEMENTS = WIDTH * HEIGHT;

    std::cout << "=== 2D Array Element-wise Multiplication on GPU ===\n";
    std::cout << "Array dimensions: " << HEIGHT << " rows x " << WIDTH << " columns\n";
    std::cout << "Total elements: " << TOTAL_ELEMENTS << "\n";

    // Initialize input arrays on the host (CPU).
    // We use nested std::array for readable 2D initialization, but we'll
    // flatten these to 1D when copying to the GPU.
    std::array<std::array<float, WIDTH>, HEIGHT> a {{
        { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f},  // Row 0
        { 8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f},  // Row 1
        {15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f},  // Row 2
        {22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f},  // Row 3
        {29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f}   // Row 4
    }};

    std::array<std::array<float, WIDTH>, HEIGHT> b {{
        { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f},
        { 8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
        {15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f},
        {22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f},
        {29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f}
    }};

    // Initialize result array on the host. We initialize to zero for
    // defensive programming, even though the kernel will overwrite everything.
    std::array<float, TOTAL_ELEMENTS> result{};

    // Print input arrays to verify our data
    print2DArray(reinterpret_cast<const float*>(a.data()), WIDTH, HEIGHT, "Input Array A");
    print2DArray(reinterpret_cast<const float*>(b.data()), WIDTH, HEIGHT, "Input Array B");

    // ========================================================================
    // GPU SETUP AND CONFIGURATION
    // ========================================================================

    // Get optimal block size based on GPU capabilities
    dim3 threadsPerBlock = getOptimalBlockSize(WIDTH, HEIGHT);

    // Calculate how many blocks we need in each dimension to cover all data.
    // This is the ceiling division formula we discussed: (N + blockSize - 1) / blockSize
    // It ensures we always have enough blocks to process every element.
    dim3 blocksPerGrid(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "Grid configuration: " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks = " << blocksPerGrid.x * blocksPerGrid.y << " total blocks\n";
    std::cout << "Total threads launched: "
              << blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y << "\n";
    std::cout << "Active threads (doing work): " << TOTAL_ELEMENTS << "\n";
    std::cout << "Thread efficiency: "
              << (100.0 * TOTAL_ELEMENTS / (blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y))
              << "%\n\n";

    // ========================================================================
    // DEVICE MEMORY ALLOCATION
    // ========================================================================

    std::cout << "Allocating device memory...\n";

    // Declare device pointers. These will hold GPU memory addresses.
    // We initialize them to nullptr for safety.
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_result = nullptr;

    // Allocate memory on the GPU for our three arrays.
    // Each allocation gets checked for errors with our CUDA_CHECK macro.
    size_t bytes = TOTAL_ELEMENTS * sizeof(float);
    std::cout << "  Allocating " << bytes << " bytes per array ("
              << bytes * 3 << " bytes total)\n";

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, bytes));

    // ========================================================================
    // DATA TRANSFER: HOST TO DEVICE
    // ========================================================================

    std::cout << "Copying data from host to device...\n";

    // Copy input data from CPU to GPU. The a.data() method returns a pointer
    // to the underlying contiguous storage, which we can copy as a flat array.
    // cudaMemcpyHostToDevice tells CUDA the direction of the transfer.
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // KERNEL LAUNCH
    // ========================================================================

    std::cout << "Launching kernel...\n";

    // Launch the kernel with our calculated grid and block dimensions.
    // The triple angle brackets <<< >>> are CUDA's syntax for kernel launches.
    // Inside, we specify: <<<blocks per grid, threads per block>>>
    scalarMult2DArray<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, WIDTH, HEIGHT);

    // Check for any errors during launch or execution.
    // This is where we catch problems like invalid grid dimensions or
    // runtime errors like out-of-bounds memory access.
    checkKernelErrors("scalarMult2DArray");

    std::cout << "Kernel completed successfully!\n";

    // ========================================================================
    // DATA TRANSFER: DEVICE TO HOST
    // ========================================================================

    std::cout << "Copying results from device to host...\n";

    // Copy the computed results back from GPU to CPU.
    // cudaMemcpyDeviceToHost specifies the direction.
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, bytes, cudaMemcpyDeviceToHost));

    // ========================================================================
    // CLEANUP
    // ========================================================================

    std::cout << "Freeing device memory...\n";

    // Free all GPU memory we allocated. It's important to free in the
    // reverse order of allocation for good practice, though CUDA doesn't
    // strictly require this.
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));

    // ========================================================================
    // RESULTS VERIFICATION
    // ========================================================================

    // Print the results in a readable 2D format
    print2DArray(result.data(), WIDTH, HEIGHT, "Result Array (A * B)");

    // Verify a few elements to make sure the computation was correct.
    // For production code, you'd want more thorough verification.
    std::cout << "\nVerification (checking a few elements):\n";
    std::cout << "  result[0] = " << result[0] << " (expected: " << a[0][0] * b[0][0] << ")\n";
    std::cout << "  result[6] = " << result[6] << " (expected: " << a[0][6] * b[0][6] << ")\n";
    std::cout << "  result[34] = " << result[34] << " (expected: " << a[4][6] * b[4][6] << ")\n";

    // Calculate and verify the sum of all results as a simple sanity check
    float sum = 0.0f;
    for (const auto& val : result) {
        sum += val;
    }
    std::cout << "  Sum of all results: " << sum << "\n";

    std::cout << "\n=== Program completed successfully! ===\n";
    return EXIT_SUCCESS;
}