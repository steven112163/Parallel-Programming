# Parallel Programming Assignment 2



###### tags: `平行程式作業`
[TOC]



## Reference
The SIMD library used in part 1 is from [link](https://github.com/lemire/SIMDxorshift).



## Q1
- **In your write-up, produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1. Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case?**
  The code below creates a block of the image for each thread. Each thread gets the starting row of the image and the number of rows it should compute.
  ![](https://i.imgur.com/lCmN8OE.png)

  The table below shows the running time and the speedup of view 1. Based on the results, it's clear that the speedup becomes nonlinear when the number of threads is three. The reason might be the locality of reference.
  |Running Time|Speedup|
  |:-:|:-:|
  |![](https://i.imgur.com/Db0BHLm.png)|![](https://i.imgur.com/Uj6UwCI.png)|



## Q2
- **How do your measurements explain the speedup graph you previously created?**

  The table below shows that the thread with ID 1 is much slower than other threads. The phenomenon confirms that the reason for the nonlinear speedup is due to the locality of reference. Therefore, the false sharing caused by locality of reference, which results from consecutive memory accesses, makes the running time slower.
  |Running Time of Each Thread|Average Running Time of Each Thread|
  |:-:|:-:|
  |![](https://i.imgur.com/Roq07wf.png)|![](https://i.imgur.com/t2v8rAJ.png)|



## Q3
- **In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.**

  The code below shows how I implemented the parallelization. Each thread calls the serial mandelbrot function with its thread ID as the starting row. Every call to the serial function only requires the computation of one row. Therefore, each thread will use "idx + number_of_threads" as the next starting row and repeat the process.
  ![](https://i.imgur.com/hkdpa9M.png)

  The results in the following table show the 4-thread speedup. In view 1, it reaches 3.79x. In view 2, it gets 3.8x.
  ||Running Time|Speedup|
  |:-:|:-:|:-:|
  |**View 1**|![](https://i.imgur.com/Tc1Me36.png)|![](https://i.imgur.com/qjI7JpF.png)|
  |**View 2**|![](https://i.imgur.com/hbn10sO.png)|![](https://i.imgur.com/8F2Cnc8.png)|



## Q4
- **Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not? (Notice that the workstation server provides 4 cores 4 threads.)**

  Based on the results in the table below, the performance isn't better. The phenomenon is because the total number of threads (or cores) is four. Therefore, only four threads can run concurrently. If the number of threads is greater than four, the CPU must perform content switching, which lowers the speedup.
  |View 1|View 2|
  |:-:|:-:|
  |![](https://i.imgur.com/qjI7JpF.png)|![](https://i.imgur.com/8F2Cnc8.png)|