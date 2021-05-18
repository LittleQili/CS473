#define NTHREADS 128

// homework1程序
// TODO: GPU版本计算两个向量差的二范数

#include <stdio.h>
#include <time.h>

__host__ int reduce_start(int *h_idata)
{
	int *d_idata;
	int *d_odata;
	int *d_intermediateSums;
	int res = 0;

	cudaError_t cudaStatus;

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
		fprintf(stderr, "cudaMemcpy failed!");
		exit(1);
	}
	// 为device分配空间及把输入拷入device的global_memory
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	for (int idx = 0; idx < 100; idx++)
	{
		dim3 dimBlock(NTHREADS, 1, 1);
		dim3 dimGrid(nBlocks, 1, 1);
		int smemSize = NTHREADS * sizeof(int);
		reduce<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, FOURMB);
		int s = nBlocks;
		while (s > 1)
		{
			dim3 dimGrid((s + NTHREADS - 1) / NTHREADS, 1, 1);
			cudaStatus = cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMemcpy failed!");
				exit(1);
			}
			reduce<<<dimGrid, dimBlock, smemSize>>>(d_intermediateSums, d_odata, s);
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
			fprintf(stderr, "cudaMemcpy failed!");
			exit(1);
		}
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("duration:%.7lfdms\n", costTime / 1000 / 1000);
	cudaFree(d_intermediateSums);
	return res;
}

int rbfComputeCPU(int *input1, int *input2, int len)
{
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	int res = 0;
	for (int i = 0; i < len; i++)
	{
		res += (input1[i] - input2[i]) * (input1[i] - input2[i]);
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	return res;
}

__host__ int main()
{
	int *h_idata1, *h_idata2;
	h_idata = (int *)malloc(1024 * 1024 * 1024 * sizeof(int));
	for (int i = 0; i < 1024 * 1024 * 1024; i++)
	{
		h_idata1[i] = rand() & 0xff;
		h_idata2[i] = rand() & 0xff;
	}
	for (int n = 1024; n <= 1024 * 1024 * 1024; n *= 4)
	{
		int cpu_result = rbfComputeCPU(h_idata1, h_idata2, n);
		int gpu_result = rbfComputeGPU(h_idata1, h_idata2, n);
		if (cpu_result != gpu_result)
		{
			printf("ERROR happen when compute %d\n", n);
			printf("cpu_result = %d,gpu_result = %d\n", cpu_result, gpu_result);
		}
	}
}