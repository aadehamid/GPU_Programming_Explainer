//
//  Created by Patricio Bulic, Davor Sluga, UL FRI on 6/6/2022.
//  Copyright © 2022 Patricio Bulic, Davor Sluga UL FRI. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 4
#define BLOCK_SIZE 16
//#define CHUNK_SIZE (BLOCK_SIZE + 2)

//***************************************************
// Image sharpening using a 3x3 kernel; 
// Source: https://setosa.io/ev/image-kernels/
//
//      |  0  -1   0 |
// K =  | -1   5  -1 |
//      |  0  -1   0 |
//
//***************************************************

__device__ inline unsigned char getIntensity(const unsigned char *image, int row, int col,
                                             int channel, int height, int width, int cpp)
{
    if (col < 0 || col >= width)
        return 0;
    if (row < 0 || row >= height)
        return 0;
    return image[(row * width + col) * cpp + channel];
}


// CUDA kernel for image sharpening. Each thread computes one output pixel
__global__ void sharpen(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height, const int cpp)
{
    // Get pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        for (int c = 0; c < cpp; c++)
        {
            unsigned char px01 = getIntensity(imageIn, y - 1, x, c, height, width, cpp);
            unsigned char px10 = getIntensity(imageIn, y, x - 1, c, height, width, cpp);
            unsigned char px11 = getIntensity(imageIn, y, x, c, height, width, cpp);
            unsigned char px12 = getIntensity(imageIn, y, x + 1, c, height, width, cpp);
            unsigned char px21 = getIntensity(imageIn, y + 1, x, c, height, width, cpp);

            short pxOut = (5 * px11 - px01 - px10 - px12 - px21);
            pxOut = MIN(pxOut, 255);
            pxOut = MAX(pxOut, 0);
            imageOut[(y * width + x) * cpp + c] = (unsigned char)pxOut;
        }
    }
}

/* SLOW
// CUDA kernel for image sharpening. Each thread computes one output pixel. Uses shared memory.
__global__ void sharpen_shared(unsigned char *imageIn, unsigned char *imageOut, int width, int height, int cpp)
{
    // Get thread Ids
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ unsigned char imgBlock[CHUNK_SIZE * CHUNK_SIZE * COLOR_CHANNELS];
    // copy image block from global memory to shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < CHUNK_SIZE * CHUNK_SIZE * COLOR_CHANNELS; i += blockDim.y * blockDim.x)
    {
        int gy = blockIdx.y * blockDim.y + ((i/COLOR_CHANNELS) / (CHUNK_SIZE)) - 1; // ROW: global Y offset + row in the thread block
        int gx = blockIdx.x * blockDim.x + ((i/COLOR_CHANNELS) % (CHUNK_SIZE)) - 1; // COLUMN: global X offset + column in the thread block
        int c = i % COLOR_CHANNELS;
        imgBlock[i] = getIntensity(imageIn, gy, gx, c, height, width, cpp);
    }
    __syncthreads();

    if (x < width && y < height)
    {
        for (int c = 0; c < cpp; c++)
        {
            int pxi = (threadIdx.y + 1) * CHUNK_SIZE + (threadIdx.x + 1);
            unsigned char px01 = imgBlock[(pxi - CHUNK_SIZE) * cpp + c];
            unsigned char px10 = imgBlock[(pxi - 1) * cpp + c];
            unsigned char px11 = imgBlock[pxi * cpp + c];
            unsigned char px12 = imgBlock[(pxi + 1) * cpp + c];
            unsigned char px21 = imgBlock[(pxi + CHUNK_SIZE) * cpp + c];

            short pxOut = (5 * px11 - px01 - px10 - px12 - px21);
            pxOut = MIN(pxOut, 255);
            pxOut = MAX(pxOut, 0);
            imageOut[(y * width + x) * cpp + c] = (unsigned char)pxOut;
        }
    }
}

*/
int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image [kernel id]\n");
        exit(EXIT_FAILURE);
    }
    
    /*
    int kernel_id=0;
    if (argc==4){
        kernel_id=atoi(argv[3]);
    }
    
    */
    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    cpp = COLOR_CHANNELS;

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Allocate device memory for images
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // Copy input image to device
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute the kernel
    cudaEventRecord(start);
    sharpen<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
    getLastCudaError("sharpen() execution failed\n");
    /*
    if(kernel_id==1){
        printf("Using shared memory kernel.\n");
        sharpen_shared<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
        getLastCudaError("sharpen_shared() execution failed\n");
    }
    else{
        printf("Using default kernel.\n");
        sharpen<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
        getLastCudaError("sharpen() execution failed\n");
    }
    */
    cudaEventRecord(stop);

    // Copy image back to host
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));

    // Wait for the event to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Retrieve output file type
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }
    // Write output image to file
    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Release device memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Clean up the two events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    // Release host memory
    free(h_imageIn);
    free(h_imageOut);

    return 0;
} 

