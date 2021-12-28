# Parallel Programming Assignment 6



###### tags: `平行程式作業`
[TOC]



==The speedup of filter 2 is not always greater than 4.0==



## Q1: Explain your implementation. How do you optimize the performance of convolution?
In the function `hostFE` below, I use `CL_MEM_COPY_HOST_PTR` to transfer data from the host to the device. After data transfer, the function executes the kernel and copy data from the device to the host.
```cpp=
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    int filter_size = filterWidth * filterWidth * sizeof(float);
    int image_size = imageWidth * imageHeight * sizeof(float);

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // Allocate device memory
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, filter_size, filter, NULL);
    cl_mem d_input_image = clCreateBuffer(*context, CL_MEM_COPY_HOST_PTR, image_size, inputImage, NULL);
    cl_mem d_output_image = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // Set arguments for the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_input_image);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_output_image);

    // Set local and global workgroups sizes
    size_t local_work_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    // Run kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);

    // Copy data from device to host
    clEnqueueReadBuffer(command_queue, d_output_image, CL_TRUE, 0, image_size, (void *) outputImage, 0,
                        NULL, NULL);

    // Release opencl object
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_input_image);
    clReleaseMemObject(d_output_image);
    clReleaseKernel(kernel);
}
```
The kernel function below computes its location in the image and calculates the convolution result.
```cpp=
__kernel void convolution(int filterWidth,
                          __constant float *filter,
                          __read_only __global float *inputImage,
                          __write_only __global float *outputImage) {
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int halfFilterSize = filterWidth >> 1;

    float sum = 0;
    int row, col;
    for (row = -halfFilterSize; row <= halfFilterSize; row++) {
        for (col = -halfFilterSize; col <= halfFilterSize; col++) {
            if (filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize] != 0 &&
                thisY + row >= 0 && thisY + row < imageHeight &&
                thisX + col >= 0 && thisX + col < imageWidth) {
                sum += inputImage[(thisY + row) * imageWidth + thisX + col] *
                filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}
```
The local size is $25 \times 25$. My implementation can accelerate the convolution.



## Q2: Rewrite the program using CUDA.
### Explain your CUDA implementation
The kernel function below resembles the OpenCL version. It computes the location and calculates the corresponding convolution result.
```cpp=
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
```
The local size in CUDA is also $25 \times 25$. However, the `hostFE` function below uses streams to hide the data transfer latency.
```cpp=
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
```
### Plot a chart to show the performance difference between using OpenCL and CUDA.
The experiments below are conducted 7 times.
- **OpenCL**
  ![](https://i.imgur.com/uV4VHNF.png)
- **CUDA**
  ![](https://i.imgur.com/ZGuny7X.png)
- **Comparison Between OpenCL and CUDA**
  The data in the graph below are the medians of the experiments and the corresponding speedup.
  ![](https://i.imgur.com/gw5jV0s.png)



### Explain the result.
From the graphs above, we can see that CUDA performs better than OpenCL. The reason might be that the GPU on the workstation is Nvidia, so that CUDA can optimize the code for the hardware. On the other hand, while OpenCL is a heterogeneous framework, it is hard for it to optimize the code for Nvidia GPU. Also, we can see that the performance of OpenCL is unstable from the experiments.