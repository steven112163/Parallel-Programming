#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 8

__global__ void mandelKernel(int *d_data,
                             int width,
                             float stepX, float stepY,
                             float lowerX, float lowerY,
                             int maxIteration) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re;
    float z_im = c_im;

    int i;
    float new_re, new_im;
    for (i = 0; i < maxIteration; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        new_re = z_re * z_re - z_im * z_im;
        new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_data[thisX + thisY * width] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    int *h_data = (int *) malloc(size);
    int *d_data;
    cudaMalloc(&d_data, size);

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(resX / NUM_THREADS, resY / NUM_THREADS);
    mandelKernel<<<grid, block>>>(d_data, resX, stepX, stepY, lowerX, lowerY, maxIterations);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    memcpy(img, h_data, size);
    cudaFree(d_data);
    free(h_data);
}
