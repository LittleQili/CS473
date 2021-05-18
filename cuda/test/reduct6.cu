#define FOURMB 2 * 1024 * 1024
#define BYTES FOURMB * sizeof(int)
#define NTHREADS 128
#define NGRIDS 256

#include <stdio.h>
#include <time.h>

template <unsigned int blockSize>
__device__ void wrapReduce(volatile int *sdata, int tid)
{
	if (blockSize >= 64)
		sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32)
		sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16)
		sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)
		sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)
		sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)
		sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n)
	{
		sdata[tid] += g_idata[i];
		if (i + blockSize < n) sdata[tid] += g_idata[i + blockSize];
		i += gridSize;
	}

	__syncthreads();

	if (blockSize >= 512 && (tid < 256))
	{
		sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256 && (tid < 128))
	{
		sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128 && (tid < 64))
	{
		sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid < 32)
		wrapReduce<blockSize>(sdata, tid);
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__host__ int reduce_start(int *h_idata)
{
	int *d_idata;
	int *d_odata;
	int *d_intermediateSums;
	int res = 0;

	cudaError_t cudaStatus;

	int nBlocks = FOURMB / NTHREADS / 2;
	cudaStatus = cudaMalloc((void **)&d_idata, BYTES);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&d_odata, nBlocks * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&d_intermediateSums, sizeof(int) * nBlocks);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

	cudaStatus = cudaMemcpy(d_idata, h_idata, BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy h_idata failed!");
		exit(1);
	}
	// 为device分配空间及把输入拷入device的global_memory
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	for (int idx = 0; idx < 100; idx++)
	{
		dim3 dimBlock(NTHREADS, 1, 1);
		dim3 dimGrid(NGRIDS,1,1);
		// dim3 dimGrid(nBlocks, 1, 1);
		int smemSize = NTHREADS * sizeof(int);
		reduce<NTHREADS><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, FOURMB);
		int s = nBlocks;
		while (s > 1)
		{
			// dim3 dimGrid((s + NTHREADS - 1) / NTHREADS, 1, 1);
			dim3 dimGrid(NGRIDS,1,1);
			cudaStatus = cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMemcpy outTointermediate failed!");
				exit(1);
			}
			reduce<NTHREADS><<<dimGrid, dimBlock, smemSize>>>(d_intermediateSums, d_odata, s);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}
			s /= (NTHREADS * 2);
		}
		cudaStatus = cudaMemcpy(&res, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaStatus = cudaGetLastError();
			fprintf(stderr, "cudaMemcpy getOutput failed! %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("duration:%.7lfdms\n", costTime / 1000 / 1000);
	cudaFree(d_intermediateSums);
	return res;
}

__host__ int main()
{
	int *h_idata;
	h_idata = (int *)malloc(BYTES);
	for (int i = 0; i < FOURMB; i++)
	{
		h_idata[i] = rand() & 0xff;
	}
	int cpu_result = 0;
	for (int i = 0; i < FOURMB; i++)
	{
		cpu_result += h_idata[i];
	}
	int gpu_result = reduce_start(h_idata);
	printf("cpu_result: %d\n", cpu_result);
	printf("gpu_result: %d\n", gpu_result);
}