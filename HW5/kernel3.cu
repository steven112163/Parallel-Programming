#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 8
#define RANGE 5

__global__ void mandelKernel(int *d_data,
                             float stepX, float stepY,
                             float lowerX, float lowerY,
                             int maxIteration,
                             int pitch,
                             int range) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * range;
    int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * range;

    float c_re, z_re;
    float c_im, z_im;
    float new_re, new_im;

    int res;
    int *row;
    for (int j = 0; j < range; j++) {
        row = (int *) ((char *) d_data + (thisY + j) * pitch);
        for (int i = 0; i < range; i++) {
            c_re = lowerX + (thisX + i) * stepX;
            c_im = lowerY + (thisY + j) * stepY;
            z_re = c_re;
            z_im = c_im;
            for (res = 0; res < maxIteration; ++res) {

                if (z_re * z_re + z_im * z_im > 4.f)
                    break;

                new_re = z_re * z_re - z_im * z_im;
                new_im = 2.f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
            }

            row[thisX + i] = res;
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    size_t pitch;
    int *h_data, *d_data;
    cudaHostAlloc(&h_data, size, cudaHostAllocMapped);
    cudaMallocPitch(&d_data, &pitch, resX * sizeof(int), resY);

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(resX / NUM_THREADS / RANGE, resY / NUM_THREADS / RANGE);
    mandelKernel<<<grid, block>>>(d_data,
                                  stepX, stepY,
                                  lowerX, lowerY,
                                  maxIterations,
                                  pitch,
                                  RANGE);

    cudaMemcpy2D(h_data, resX * sizeof(int), d_data, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_data, size);
    cudaFreeHost(h_data);
    cudaFree(d_data);
}
