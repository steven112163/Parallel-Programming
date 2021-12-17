#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define LOCAL_SIZE 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    cl_int status;
    int filter_size = filterWidth * filterWidth * sizeof(float);
    int image_size = imageWidth * imageHeight * sizeof(float);

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    // Allocate device memory
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_size, NULL, &status);
    cl_mem d_input_image = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size, NULL, &status);
    cl_mem d_output_image = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, &status);

    // Copy data from host to device
    status = clEnqueueWriteBuffer(command_queue, d_filter, CL_TRUE, 0, filter_size, (void *) filter, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, d_input_image, CL_TRUE, 0, image_size, (void *) inputImage, 0, NULL,
                                  NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", status);

    // Set arguments for the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &filterWidth);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_filter);
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &imageHeight);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &imageWidth);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &d_input_image);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &d_output_image);

    // Set local and global workgroups sizes
    size_t local_work_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    // Run kernel
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);

    // Copy data from device to host
    status = clEnqueueReadBuffer(command_queue, d_output_image, CL_TRUE, 0, image_size, (void *) outputImage, NULL,
                                 NULL, NULL);

    // release opencl object
    status = clReleaseCommandQueue(command_queue);
    status = clReleaseMemObject(d_filter);
    status = clReleaseMemObject(d_input_image);
    status = clReleaseMemObject(d_output_image);
    status = clReleaseKernel(kernel);
}