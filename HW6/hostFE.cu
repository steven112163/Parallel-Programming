#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hostFE.h"
}

#define NUM_THREADS 25
#define NUM_STREAMS 4

__global__ void convKernel(int filter_width,
                           float *filter,
                           int offset,
                           int image_height,
                           int image_width,
                           float *input_image,
                           float *output_image) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y + offset;

    int half_filter_size = filter_width >> 1;
    float sum = 0.0f;
    int row, col;
    for (row = -half_filter_size; row <= half_filter_size; row++) {
        for (col = -half_filter_size; col <= half_filter_size; col++) {
            if (thisY + row >= 0 && thisY + row < image_height &&
                thisX + col >= 0 && thisX + col < image_width) {
                sum += input_image[(thisY + row) * image_width + thisX + col] *
                       filter[(row + half_filter_size) * filter_width + col + half_filter_size];
            }
        }
    }
    output_image[thisY * image_width + thisX] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    // Allocate device memory
    int filter_size = filterWidth * filterWidth * sizeof(float);
    int image_size = imageWidth * imageHeight * sizeof(float);
    float *d_filter, *d_input_image, *d_output_image;
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_input_image, image_size);
    cudaMalloc(&d_output_image, image_size);

    // Pin outputImage
    cudaHostRegister(outputImage, image_size, cudaHostRegisterPortable);

    // Copy filter and input_image from host to device
    cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_image, inputImage, image_size, cudaMemcpyHostToDevice);

    // Setup streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    int grid_step = imageHeight / NUM_STREAMS;
    int grid_size = image_size / NUM_STREAMS;

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(imageWidth / NUM_THREADS, grid_step / NUM_THREADS);
    int offset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        convKernel<<<grid, block, 0, streams[i]>>>(filterWidth, d_filter, offset, imageHeight, imageWidth,
                                                   d_input_image, d_output_image);
        cudaMemcpyAsync(outputImage + imageWidth * offset, d_output_image + imageWidth * offset, grid_size,
                        cudaMemcpyDeviceToHost, streams[i]);
        offset += grid_step;
    }

    // Free memory
    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaHostUnregister(outputImage);
    cudaFree(d_filter);
    cudaFree(d_input_image);
    cudaFree(d_output_image);
}
