#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hostFE.h"
}

#define NUM_THREADS 25

__global__ void convKernel(int filter_width,
                           float *filter,
                           int image_height,
                           int image_width,
                           float *input_image,
                           float *output_image) {
    const int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    int half_filter_size = filter_width / 2;
    float sum = 0.0f;
    int row, col;
    for (row = -half_filter_size; row < half_filter_size + 1; row++) {
        for (col = -half_filter_size; col < half_filter_size + 1; col++) {
            if (coord.y + row > -1 && coord.y + row < image_height &&
                coord.x + col > -1 && coord.x + col < image_width) {
                sum += input_image[(coord.y + row) * image_width + coord.x + col] *
                       filter[(row + half_filter_size) * filter_width + col + half_filter_size];
            }
        }
    }
    output_image[coord.y * image_width + coord.x] = sum;
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

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(imageWidth / NUM_THREADS, imageHeight / NUM_THREADS);
    convKernel<<<grid, block>>>(filterWidth, d_filter, imageHeight, imageWidth, d_input_image, d_output_image);

    // Copy output_image from device to host
    cudaDeviceSynchronize();
    cudaMemcpy(outputImage, d_output_image, image_size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaHostUnregister(outputImage);
    cudaFree(d_filter);
    cudaFree(d_input_image);
    cudaFree(d_output_image);
}
