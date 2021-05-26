// #define DEBUG
#ifdef DEBUG
#define FOURMB (2 * 1024 * 1024)
#define BYTES (FOURMB * sizeof(int))
#define NTHREADS 128
#define INITN 256
#else
#define FOURMB (2 * 1024 * 1024)
// #define FOURM
#define BYTES (FOURMB * sizeof(int))
#define NTHREADS 128
#define INITN 1024
#endif

// homework1程序
// TODO: GPU版本计算两个向量差的二范数

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define CUDA_CALL(x)                                                               \
	{                                                                              \
		const cudaError_t a = (x);                                                 \
		if (a != cudaSuccess)                                                      \
		{                                                                          \
			printf("\nCUDA Error: %s (err_num = %d)\n", cudaGetErrorString(a), a); \
			cudaDeviceReset();                                                     \
			assert(0);                                                             \
		}                                                                          \
	}

// TODO: 定义GPU kernel函数,并在rbfComputeGPU中调用

__device__ void wrapReduce(volatile int *sdata, int tid)
{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	sdata[tid] = (i < n) ? g_idata[i] : 0;
	sdata[tid] += (i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0;

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid < 32)
		wrapReduce(sdata, tid);
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void norm2(int *input, int *output, int len)
{
	extern __shared__ int smem[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	smem[tid] = input[(i<<1)] - input[(i<<1)+1];
	__syncthreads();
	smem[tid] = smem[tid] * smem[tid];
	output[i] = smem[tid];
}

// no fusion version:
__host__ int rbfComputeGPU(int *input1, int *input2, int len)
{
	int *d_idata1;
	// int *d_idata2;
	int *d_idata;
	int *d_odata;
	int *d_intermediateSums;
	int res = 0;

	int nReduBlocks = len / NTHREADS / 2;
	int n2NormBlocks = len / NTHREADS;
	int calBytes = len * sizeof(int) * 2;

	// TODO: 在gpu上分配空间
	// CUDA_CALL();
	CUDA_CALL(cudaMalloc((void **)&d_idata1, calBytes));
	// CUDA_CALL(cudaMalloc((void **)&d_idata2, calBytes));
	CUDA_CALL(cudaMalloc((void **)&d_idata, calBytes / 2));
	CUDA_CALL(cudaMalloc((void **)&d_odata, nReduBlocks * sizeof(int)));
	CUDA_CALL(cudaMalloc((void **)&d_intermediateSums, sizeof(int) * nReduBlocks));
	// TODO: 将cpu的输入拷到gpu上的globalMemory
	for (int i = 0; i < len; ++i)
	{
		CUDA_CALL(cudaMemcpy(&d_idata1[i * 2], &input1[i], sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(&d_idata1[i * 2+1], &input2[i], sizeof(int), cudaMemcpyHostToDevice));
	}
	// CUDA_CALL(cudaMemcpy(d_idata1, input1, calBytes, cudaMemcpyHostToDevice));
	// CUDA_CALL(cudaMemcpy(d_idata2, input2, calBytes, cudaMemcpyHostToDevice));

#ifdef DEBUG
	int *test2norm;
	test2norm = (int *)malloc(calBytes);
	assert(test2norm);
#endif

	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	// 重复100次,比较时间
	for (int idx = 0; idx < 100; idx++)
	{
		// res = 0;
		// TODO: 启动gpu计算函数,计算两个向量间的RBF
		dim3 dimBlock(NTHREADS, 1, 1);
		dim3 dimGrid(n2NormBlocks, 1, 1);
		int smemSize = NTHREADS * sizeof(int);
		norm2<<<dimGrid, dimBlock, smemSize>>>(d_idata1, d_idata, len);
#ifdef DEBUG
		CUDA_CALL(cudaMemcpy(test2norm, d_idata, calBytes, cudaMemcpyDeviceToHost));
#endif
		dimGrid.x = nReduBlocks;
		reduce<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, len);
		int s = nReduBlocks;
		while (s > 1)
		{
			dim3 dimGrid((s + NTHREADS - 1) / NTHREADS, 1, 1);
			CUDA_CALL(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice));
			reduce<<<dimGrid, dimBlock, smemSize>>>(d_intermediateSums, d_odata, s);
			CUDA_CALL(cudaGetLastError());
			s /= (NTHREADS * 2);
		}
		// TODO: 将gpu的输出拷回cpu
		CUDA_CALL(cudaMemcpy(&res, d_odata, sizeof(int), cudaMemcpyDeviceToHost));
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("GPU cal %d size cost:%.7lfms\n", len, costTime / 1000 / 1000);
	// TODO: 释放掉申请的gpu内存
#ifdef DEBUG
	printf("test for 2norm in GPU:\n");
	for (int i = 0; i < len; ++i)
	{
		if ((input1[i] - input2[i]) * (input1[i] - input2[i]) != test2norm[i])
			printf("i:%d, test:%d, true:%d\n", i, test2norm[i], (input1[i] - input2[i]) * (input1[i] - input2[i]));
	}
	free(test2norm);
#endif
	// CUDA_CALL(cudaFree(d_idata2));
	CUDA_CALL(cudaFree(d_idata1));
	CUDA_CALL(cudaFree(d_idata));
	CUDA_CALL(cudaFree(d_odata));
	CUDA_CALL(cudaFree(d_intermediateSums));
	return res;
}

// cpu版本
int rbfComputeCPU(int *input1, int *input2, int len)
{
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	int res = 0;
	for (int idx = 0; idx < 100; idx++)
	{
		res = 0;
		for (int i = 0; i < len; i++)
		{
			res += (input1[i] - input2[i]) * (input1[i] - input2[i]);
		}
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("CPU cal %d size cost:%.7lfms\n", len, costTime / 1000 / 1000);
	return res;
}

__host__ int main()
{
	int *h_idata1, *h_idata2;
	h_idata1 = (int *)malloc(BYTES);
	h_idata2 = (int *)malloc(BYTES);
	assert(h_idata1);
	assert(h_idata2);
	srand((unsigned)time(NULL));
	for (int i = 0; i < FOURMB; i++)
	{
		h_idata1[i] = rand() & 0xff;
		h_idata2[i] = rand() & 0xff;
	}
	printf("initialize ready\n");
	for (int n = INITN; n <= FOURMB; n *= 4)
	{
		printf("n=%d:\n", n);
		int cpu_result = rbfComputeCPU(h_idata1, h_idata2, n);
		int gpu_result = rbfComputeGPU(h_idata1, h_idata2, n);
		if (cpu_result != gpu_result)
		{
			printf("ERROR happen when compute %d\n", n);
			printf("cpu_result = %d,gpu_result = %d\n", cpu_result, gpu_result);
			free(h_idata1);
			free(h_idata2);
			exit(1);
		}
	}
	free(h_idata1);
	free(h_idata2);
}