#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BLOCKS 8
#define NUM_STREAMS 10

__global__ void mandelKernel(int *d_data,
                             int width, int offset,
                             float stepX, float stepY,
                             float lowerX, float lowerY,
                             int maxIteration) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y + offset);

    float2 c = make_float2(lowerX + coord.x * stepX, lowerY + coord.y * stepY);
    float2 z = c;
    float2 new_z;

    int i;
    for (i = 0; i < maxIteration; i++) {
        if (z.x * z.x + z.y * z.y > 4.f)
            break;

        new_z.x = z.x * z.x - z.y * z.y;
        new_z.y = 2.f * z.x * z.y;
        z.x = c.x + new_z.x;
        z.y = c.y + new_z.y;
    }

    d_data[coord.x + coord.y * width] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    int *d_data;
    cudaMalloc(&d_data, size);
    cudaHostRegister(img, size, cudaHostRegisterPortable);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    int grid_step = resY / NUM_STREAMS;
    int grid_size = size / NUM_STREAMS;

    dim3 block(NUM_BLOCKS, NUM_BLOCKS);
    dim3 grid(resX / NUM_BLOCKS, grid_step / NUM_BLOCKS);
    int offset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        mandelKernel<<<grid, block, 0, streams[i]>>>(d_data, resX, offset, stepX, stepY, lowerX, lowerY, maxIterations);
        cudaMemcpyAsync(img + resX * offset, d_data + resX * offset, grid_size, cudaMemcpyDeviceToHost, streams[i]);
        offset += grid_step;
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaHostUnregister(img);
    cudaFree(d_data);
}
