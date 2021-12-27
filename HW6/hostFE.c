#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define LOCAL_SIZE 25

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    int filter_size = filterWidth * filterWidth * sizeof(float);
    int image_size = imageWidth * imageHeight * sizeof(float);
    int halfFilterSize = filterWidth >> 1;

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
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *) &halfFilterSize);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_filter);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_input_image);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &d_output_image);

    // Set local and global workgroups sizes
    size_t local_work_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    // Run kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);

    // Copy data from device to host
    clEnqueueReadBuffer(command_queue, d_output_image, CL_TRUE, 0, image_size, (void *) outputImage, 0,
                        NULL, NULL);

    // release opencl object
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_input_image);
    clReleaseMemObject(d_output_image);
    clReleaseKernel(kernel);
}