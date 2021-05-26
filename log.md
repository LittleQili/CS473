# Log of CS473 HW
加油，提高效率，按时休息，一定可以写完。
## CUDA 编程部分
### 作业思路
- 或许可以用shared memory解决一下。
加油！
- 再学习一下reduction的解决流程。这个问题是两个数组，基本的解决思想是差不多的。**处理线性区域**

  学到的思路：
  - 线程尽量操作线性区域，不要跳块。这样能尽可能用满内存的带宽（优化2）
  - 可以合并不同block（优化3）、不同grid（优化6）之间的global memory访问。
  - loop unrolling: 1)（优化4）利用SIMT的性质，展开warp内的循环，减少syncthreads操作；2)（优化5）利用block size的一个维度最多1024个线程，完成循环的完全展开
- 先读一下代码，读一下读一下
- 大致思路：
  1. 2 kernel: 先完成两个向量的剑法、平方，然后采用reduce的kernel进行reduce
  2. Fusion 上面两个kernel.

- 接下来的工作：
  1. 想一想如何fusion
  2. 总结一下这6种优化方法目的和效果的关系，整理一下运行效果，写个报告出来。

### 遇到的问题与解决

- 在进行地址传递的时候，分配到device里面的地址只需要一层地址传递就行。不需要把这个地址外面的壳子再套一层进行传递，这样会出问题（虽然这里还是不太懂0.0）
- **未解决**：mm_shared_mem.cu里面为什么需要第二个`__syncthreads()`?
- **未完全理解**：在reduction的一系列例子中，是不是优化2（笔记）能够减少内存访问的冲突。
- **未解决**：当数组数并非2次方，应该怎样对齐？

### 知识点
#### 2.基本概念: Grids, Blocks, Threads

##### 有关kernel的写法汇总
- kernel里面读取的内置变量
  - threadIdx: 当前block内部thread的ID
  - blockIdx: 当前grid内部block的ID
  - blockDim: 当前block里面各个维度有多少个thread
  - gridDim: 当前grid里面各个维度有多少个block
  - warpSize: 一个SM里面最多跑几个线程, 事实上 = 32

- kernel 调用相关
  - `kernelfunc<<<dimGrid, dimBlock>>>`

    `dimGrid`指的是一个grid里面，block的构成，比如dimGrid = (2,2)表示有4个block两个维度，每个维度上有两个

    `dimBlock`指的是一个block里面，thread的构成，比如dimBlock = (32,32)表示有32*32个thread两个维度，每个维度上有32个

    `dimGrid` `dimBlock` 可以使用`dim3`定义。
  - **重要**
    
    CUDA的thread、block 用内置变量获取方式和内存模型完全不同。视觉上的横向是x（内存是y）。所以kernel function代码会出现这样的情况：
    ```c
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    ```
  - CUDA kernel调用相关的各个常量的上限(好像也包含cache) [ref](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability)

    这里列出几个常用的：
    - in-grid: x:(2\^31-1), y: 65535, z: 65535
    - in-block: x: 1024, y: 1024, z: 64

  

##### memory相关
- `__syncthreads()`
  
  是**block内**thread同步的工具。具体可以参考p136官方文档。
  > waits until all threads in the thread block have reached this point and all global and shared
  memory accesses made by these threads prior to __syncthreads() are visible to all threads
  in the block.
- host 和 device memory 在DRAM中被认为是完全分开的。Kernel 在 device memory上运行。
- Linear memory: 其实就是我们平常说的一块内存用指针访问。
  - `cudaMalloc()`: 在global memory中分配空间
    ```c
    float* d_A;cudaMalloc(&d_A, size);
    ```
  - `cudaMallocPitch()`
    ```c
    // devPtr: 相当于整体的指针；pitch:记录一行有多大
    float* devPtr; size_t pitch;
    cudaMallocPitch(&devPtr,&pitch,width*sizeof(float),height);
    // in Device:
    for(int r = 0;r < height;++r){
      float* row = (float*)((char*)devPtr + r*pitch);
      for(int c = 0;c < width;++c) row[c];
    }
    ```
  - `cudaMalloc3D()`

    具体参考Nvidia书的第22页

  - ? `cudaMemcpyToSymbol()`


##### 单SM多warp的硬件流程

- 如果没有指令执行，就发射一个warp的指令。
- 如果一个warp中所有线程被Suspended（阻塞，书上指因为访存），那么发射下一个warp的指令
- 线程执行完毕自动消亡
- 会整合访存请求，并且按序发回

#####  线程块调度（书上的例子，不完整版）

暂时看了一下直方图的例子。学习到：

- 原子操作

- 每个warp(半个warp)可以自动合并访存请求

- 每次读取应该尽量占满内存带宽的最低存储大小（？）

- 采用shared memory，可以整合写回global memory

  > syncthreads()这个是同步Block，SM还是warp?


##### 分支（？）

后面再看

##### 具体调度过程中，SM warp block的关系和区别

#### x.一些零散细节

##### 异常处理

可以用宏定义的方式简化这个过程。建议对所有cuda api都使用这个操作。

```c
#define CUDA_CALL(x) {const cudaError_t a = (x); if(a != cudaSuccess){ printf("\nCUDA Error: %s (err_num = %d)\n", cudaGetErrorString(a),a); cudaDeviceReset(); assert(0);}}
```

对于核函数，使用另一种异常检测的方式，在核函数调用之后紧接着调用这个函数：

```c
__host__ void cuda_error_check(const char *prefix, const char *postfix){
    if (cudaPeekAtLastError()!= cudaSuccess){
        printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        cudaDeviceReset();
        wait_exit();
        exit(1);
	}
}
```

一个调用示例：

```c
const_test_gpu_const <<<num_blocks, num_threads >>>(data_gpu, num_elements);
cuda_error_check("Error ", " returned from costant startup kernel");
```
