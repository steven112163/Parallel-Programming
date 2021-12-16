# Parallel Programming Assignment 5



###### tags: `平行程式作業`
[TOC]



## Q1: What are the pros and cons of the three methods? Give an assumption about their performances.
### Method 1
- **Pros**
   - Each thread only needs to calculate one pixel.
   - The host memory is pageable, so the system performance won't be reduced.
- **Cons**
   - Some threads might need to wait for slower threads.
   - The host memory is not pinned, and it makes data transfer slower.
### Method 2
- **Pros**
   - Each thread only needs to calculate one pixel.
   - The host memory is pinned, so it makes data transer faster.
   - `cudaMallocPitch` pads 2D matrix with extra bytes to make each row properly aligned such that memory transactions will be faster.
- **Cons**
   - Some threads might need to wait for slower threads.
   - Cause the host memory is not pageable, it might reduce system performance.
   - The extra bytes padded by `cudaMallocPitch` make the program consume larger memory.
### Method 3
- **Pros**
   - Each thread calculates a region of pixels, which lowers the total number of threads. Also, it might make the workload of each thread average.
   - The host memory is pinned, so it makes data transer faster.
   - `cudaMallocPitch` pads 2D matrix with extra bytes to make each row properly aligned such that memory transactions will be faster.
- **Cons**
   - The workload of each thread is higher, which might make the slower threads much slower.
   - Calculating a region, which is a serial code, in each thread will lower the throughput.
   - Cause the host memory is not pageable, it might reduce system performance.
   - The extra bytes padded by `cudaMallocPitch` make the program consume larger memory.
### Assumption
The differences between these three methods are memory allocation and workload assignment. Between method 1 and 2, method 1 uses smaller memory than method2. Between method 2 and 3, method 2 assigns a pixel to each thread instead of a region of pixels. Based on the aformentioned reasions, I think smaller memory usage will make method 1 faster than method 2, and light workload will make method 2 faster than method 3.
$\Rightarrow$ method 1 < method 2 < method 3 (in computation time)



## Q2: How are the performances of the three methods? Plot a chart to show the differences among the three methods
The experiments below are executed 10 times, and the average of the results is shown in the graphs. Each thread in the kernel 3 computes 25 pixels.
### For VIEW 1 and VIEW 2
- **`maxIteration` = 1000**
  ![](https://i.imgur.com/JZYavda.png)
- **`maxIteration` = 10000**
  ![](https://i.imgur.com/ck8uTXB.png)
- **`maxIteration` = 100000**
  ![](https://i.imgur.com/dR36DcI.png)

### For different maxIteration (1000, 10000, and 100000).
- **View 1**
  ![](https://i.imgur.com/jYx3G4B.png)
- **View 2**
  ![](https://i.imgur.com/QO63C6x.png)

### Experimental Results
All results are in milliseconds.
<style>
    td, th {
        text-align: center;
    }
</style>
<table style="width:100%;">
    <thead>
        <th colspan="5">View 1</th>
    </thead>
    <tr>
        <td colspan="2" rowspan="2"></td>
        <th colspan="3">Method</th>
    </tr>
    <tr>
        <th>1</th>
        <th>2</th>
        <th>3</th>
    </tr>
    <tr>
        <th rowspan="3">Iterations</th>
        <th>1000</th>
        <td>7.000</td>
        <td>8.932</td>
        <td>10.305</td>
    </tr>
    <tr>
        <th>10000</th>
        <td>33.179</td>
        <td>35.395</td>
        <td>60.968</td>
    </tr>
    <tr>
        <th>100000</th>
        <td>304.296</td>
        <td>308.466</td>
        <td>470.763</td>
    </tr>
</table>
<table>
    <thead>
        <th colspan="5">View 2</th>
    </thead>
    <tr>
        <td colspan="2" rowspan="2"></td>
        <th colspan="3">Method</th>
    </tr>
    <tr>
        <th>1</th>
        <th>2</th>
        <th>3</th>
    </tr>
    <tr>
        <th rowspan="3">Iterations</th>
        <th>1000</th>
        <td>4.108</td>
        <td>6.803</td>
        <td>8.088</td>
    </tr>
    <tr>
        <th>10000</th>
        <td>6.918</td>
        <td>9.428</td>
        <td>18.018</td>
    </tr>
    <tr>
        <th>100000</th>
        <td>28.358</td>
        <td>30.754</td>
        <td>113.546</td>
    </tr>
</table>



## Q3: Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.
### View 1 `nvprof` Results
- **Kernel 1**
  ![](https://i.imgur.com/UE36saX.png)
- **Kernel 2**
  ![](https://i.imgur.com/JGn0JFt.png)
- **Kernel 3**
  ![](https://i.imgur.com/k4Tpcwt.png)

Based on the above graphs and tables, we can see that the performance matches my assumption. From the `nvprof` results of kernels 1 and 2, the execution time of GPU is similar because of the single pixel calculation. While from the execution time of API calls, kernel 1 is faster than kernel 2 because of smaller memory usage. From the `nvprof` results of kernels 2 and 3, the execution time of GPU shows that kernel 2 is faster than kernel 3 because of the single pixel calculation. Overall, the results confirm that kernel 1 is the fastest, and kernel 3 is the slowest.

### Kernel 3 `nvprof` Results
- **View 1**
  ![](https://i.imgur.com/k4Tpcwt.png)
- **View 2**
  ![](https://i.imgur.com/f4q14MW.png)

From the two above graphs, we can see that the workload between each thread is much closer in view 2, which is expected because the workload is much balanced in view 2 compared to view 1 from the graphs below.
![](https://i.imgur.com/wxrMgKX.png)



## Q4: Can we do even better? Think a better approach and explain it. Implement your method in kernel4.cu.
According to the above results, I chose to improve kernel 1 as the new kernel 4. In the kernel function, I changed most variables to vector type.
```cpp=
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
```

In the host function, I directly copy data from the device to the output array in the host instead of copying from the device to the host memory, then to the output array. In addition, I used streams to hide the latency of data transfer.
```cpp=
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
```