# Parallel Programming Assignment 4



###### tags: `平行程式作業`
[TOC]



## Part 1
### Q1
1. **How do you control the number of MPI processes on each node?**
    1. Put `slot=#` after hostnames in the hostfile
    2. Run the program like `mpirun -H aa -np 1 program : -H bb,cc -np 2 program`. It will make host aa run the program with 1 process, and host bb and cc run the program with 2 processes respectively.
    3. Use `--map-by` to map each process to each node, slot, etc.
2. **Which functions do you use for retrieving the rank of an MPI process and the total number of processes?**
   `int MPI_Comm_rank ( MPI_Comm comm, int *rank )` is used for retrieving the rank of an MPI process, and `int MPI_Comm_size ( MPI_Comm comm, int *size )` is used to retrieve the total number of processes.

### Q2
1. **Why MPI_Send and MPI_Recv are called “blocking” communication?**
   They are called "blocking" is because they both wait until the data transfer finished. If the data size is small, MPI_Send blocks until transferring the data into the system buffer of receiving node is complete. If the data size is large, it has to block until the data transfer to the user's buffer is complete. MPI_Recv blocks until the data is copied into the user's buffer.
2. **Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.742710|
   |4|3.544033|
   |8|1.730747|
   |12|1.172859|
   |16|0.923084|

   ![](https://i.imgur.com/K7z9OAF.png)

### Q3
1. **Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.777657|
   |4|3.422625|
   |8|1.803603|
   |16|0.941322|

   ![](https://i.imgur.com/Z8FbbFd.png)
2. **How does the performance of binary tree reduction compare to the performance of linear reduction?**
   Based on the measurements, their performances are close.
3. **Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance of Linear Reduction (s)|Performance of Binary Tree Reduction (s)|
   |:-:|:-:|:-:|
   |2|6.742710|6.777657|
   |4|3.544033|3.422625|
   |8|1.730747|1.803603|
   |16|0.923084|0.941322|
   |32|0.524869|0.480484|

   ![](https://i.imgur.com/fvaxYoR.png)
   According to the result, their performances are still close. However, there is a slight difference during the summation process. In linear reduction, all processes send their results to process 0, which needs $process - 1$ data transfer times. In contrast, binary tree reduction needs $2 \times (process - 1)$ data transfer times. Whereas we can not see that the linear reduction performs significantly better than the binary tree reduction in the above graph. It might be because the number of processes is not large enough or the transferred data is not large enough.

### Q4
1. **Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.799243|
   |4|3.768059|
   |8|1.723564|
   |12|1.169002|
   |16|1.002430|

   ![](https://i.imgur.com/Aqv7KXR.png)
2. **What are the MPI functions for non-blocking communication?**
    - **Send**
        - **MPI_Isend**: standard-mode nonblocking send
        - **MPI_Issend**: nonblocking synchronous send
        - **MPI_Irsend**: ready-mode nonblocking send
        - **MPI_Ibsend**: nonblocking buffered send
    - **Receive**
        - **MPI_Irecv**: standard-mode nonblocking receive
        - **MPI_Imrecv**: non-blocking receive for a matched message
3. **How the performance of non-blocking communication compares to the performance of blocking communication?**
   |# of MPI Processes|Blocking Performance (s)|Non-Blocking Performance (s)|
   |:-:|:-:|:-:|
   |2|6.742710|6.799243|
   |4|3.544033|3.768059|
   |8|1.730747|1.723564|
   |12|1.172859|1.169002|
   |16|0.923084|1.002430|

   ![](https://i.imgur.com/s9uGWtA.png)
   According to the graph above, their performances are close. It might be because there is no extra computation between receive and wait on the master. Thus, we can not see a considerable performance improvement.

### Q5
1. **Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.667590|
   |4|3.554108|
   |8|1.784533|
   |12|1.370821|
   |16|1.002686|

   ![](https://i.imgur.com/y1U1ZqL.png)

### Q6
1. **Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.761534|
   |4|3.475385|
   |8|1.778904|
   |12|1.164414|
   |16|0.971082|

   ![](https://i.imgur.com/mRQJ7Ww.png)

### Q7
1. **Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.**
   The measurement is conducted 10 times.
   |# of MPI Processes|Performance (s)|
   |:-:|:-:|
   |2|6.674785|
   |4|3.530522|
   |8|1.745863|
   |12|1.197773|
   |16|0.925097|

   ![](https://i.imgur.com/HxXTq48.png)
2. **Which approach gives the best performance among the 1.2.1-1.2.6 cases? What is the reason for that?**
   One-sided communication is the best approach because the number of messages in its communication is the same as linear reduction. However, its sending and receiving is non-blocking. Therefore, workers can directly write their results into the buffer of the master.

### Q8
1. **Plot ping-pong time in function of the message size for cases 1 and 2, respectively.**
    - **Case 1**
      ![](https://i.imgur.com/mpTgWsk.png)
    - **Case 2**
      ![](https://i.imgur.com/QPA2RX9.png)
2. **Calculate the bandwidth and latency for cases 1 and 2, respectively.**
    - **Case 1**
        - $T(n) = 1.52104146 \times 10^{-10} n + 3.18012197 \times 10^{-5}$
        - $\Rightarrow \ bandwidth = \frac{1}{1.52104146 \times 10^{-10}} = 6.57444275056 \times 10^{9} = 6.574 \ GB/s$
        - $\Rightarrow \ latency = 3.18012197 \times 10^{-5} = 0.0318 \ ms$
    - **Case 2**
        - $T(n) = 8.59101630 \times 10^{-9} n + 6.15770777 \times 10^{-4}$
        - $\Rightarrow \ bandwidth = \frac{1}{8.59101630 \times 10^{-9}} = 0.116400663796 \times 10^{9} = 0.116 \ GB/s$
        - $\Rightarrow \ latency = 6.15770777 \times 10^{-4} = 0.6158 \ ms$



## Part 2
### Q9
1. **Describe what approach(es) were used in your MPI matrix multiplication for each data set.**
   ![](https://i.imgur.com/1shonWT.png)
   The master process reads variables n, m, and l. Then, it uses *non-blocking send* to send them to the worker processes. Worker processes use *standard receive* to get the variables. Once these variables are received, a process can construct matrices a and b.
   ![](https://i.imgur.com/brt3nrG.png)
   Every process needs to calculate the corresponding region of the matrix result to each worker process. Each region contains multiple rows.
   ![](https://i.imgur.com/gdN7eBX.png)
   ![](https://i.imgur.com/eSaV9KC.png)
   After calculating regions, the master process reads matrices a and b. Then, it uses *non-blocking send* to send the corresponding region of matrix a to each worker process. Next, worker processes use *standard receive* to retrieve their region and put it into the corresponding region of matrix a. While matrix b is transferred intact.
   ![](https://i.imgur.com/WSdL1Zz.png)
   After transferring matrices a and b, the master process creates the matrix result and a window for retriving the products.
   ![](https://i.imgur.com/vPJcWc2.png)
   Each worker process multiplies matrix a with matrix b and puts the product into matrix result. The multiplication is split into blocks to be cache-friendly. Also, the loop of middle_idx is not the innermost loop for the same reason.
   ![](https://i.imgur.com/5PJsvG8.png)
   After multiplication, each worker process sends their corresponding region of matrix result back to the master process through the created window.
   ![](https://i.imgur.com/A7GFzIz.png)
   Finally, the master process prints the matrix result after all worker processes have finished multiplication.