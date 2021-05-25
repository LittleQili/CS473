#define NTHREADS 128

// homework1程序
// TODO: GPU版本计算两个向量差的二范数

#include <stdio.h>
#include <time.h>

// TODO: 定义GPU kernel函数,并在rbfComputeGPU中调用

__host__ int rbfComputeGPU(int *input1, int *input2, int len)
{
	int *d_idata1;
	int *d_idata2;
	int *d_odata;

	// TODO: 在gpu上分配空间
	// TODO: 将cpu的输入拷到gpu上的globalMemory
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	// 重复100次,比较时间
	for (int idx = 0; idx < 100; idx++)
	{
		// TODO: 启动gpu计算函数,计算两个向量间的RBF

		// TODO: 将gpu的输出拷回cpu
		cudaDeviceSynchronize();
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("GPU cal %d size cost:%.7lfms\n", len, costTime / 1000 / 1000);
	// TODO: 释放掉申请的gpu内存

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
	h_idata1 = (int *)malloc(4 * 1024 * 1024 * sizeof(int));
	h_idata2 = (int *)malloc(4 *1024 * 1024 * sizeof(int));
	for (int i = 0; i < 4 * 1024 * 1024; i++)
	{
		h_idata1[i] = rand() & 0xff;
		h_idata2[i] = rand() & 0xff;
	}
	printf("initialize ready\n");
	for (int n = 1024; n <=  4 * 1024 * 1024; n *= 4)
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