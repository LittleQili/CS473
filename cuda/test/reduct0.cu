#define POW21 2 * 1024 * 1024
#define POW14 16 * 1024
#define POW21BYTES POW21 * sizeof(int)
#define NTHREADS 128

#include <stdio.h>
#include <time.h>

// 使用128个线程,将分配到的128个数累加到1个数上
// 并存到 g_odata[blockIdx.x]
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{
	__shared__ int sdata[128];
	// 在shared_memory中开辟一个128int的空间

	// threadIdx.x 为线程在当前块中的位置
	unsigned int tid = threadIdx.x; // 0 - 127
	// threadIdx.x 为线程在当前块中的位置
	// blockIdx.x 为当前块ID
	// blockDim.x 为块中线程数,在当前程序中恒定为128
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 
	// 可以得到当前线程在所有线程中的编号

	sdata[tid] = g_idata[i];
	// 将global_memory中的数读取到shared_memory中

	__syncthreads();
	// 同步块中所有线程的进度,等到块中所有线程都到达此处后才会继续进行

	// blockDim.x 恒为128
	// stride 1,2,4,8,16,32,64
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;
		if(index<blockDim.x){
			sdata[index] += sdata[index+stride];
		}
		// if ((tid % (2 * stride)) == 0)
		// {
		// 	sdata[tid] += sdata[tid + stride]; 
		// }
		// stride为1时,当前块中为0 2 4 6 8 10 ... 的线程执行 sdata[tid] += sdata[tid + 1],对应图中第一步
		// stride为2时,当前块中为0 4 8 12 16 20 ... 的线程执行 sdata[tid] += sdata[tid + 2],对于图中第二步
		__syncthreads();
		// 同步所有线程的进度,等到所有的线程都完成第一步才能走第二步
	}

	// 使用块中第一个线程,将shared_memory中的和写回global_memory
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	// 此步完成后
	// g_odata[blockIdx]的值等于 第blockIdx个块对应的128个数据的和
	// g_odata[blockIdx.x] = g_odata[blockIdx.x*128]+g_odata[blockIdx.x*128+1]+...+g_odata[blockIdx.x*128+127]
}

__host__ int reduce_start(int *h_idata)
{
	int *d_idata;
	int *d_odata;
	int *d_intermediateSums;
	int res = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void **)&d_idata, POW21BYTES); // 在global_memory中分配2^21个int的空间,容纳cpu传来的输入
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&d_odata, POW14 * sizeof(int)); // 开辟2^14个int的空间,容纳第一次的reduce的输出
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&d_intermediateSums, sizeof(int) * POW14); // 开辟2^14个int的输出,容纳第一次的reduce的输出
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}
	cudaStatus = cudaMemcpy(d_idata, h_idata, POW21BYTES, cudaMemcpyHostToDevice); // 将cpu的输入传给device之前开辟的空间
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		exit(1);
	}
	// 开始计时
	struct timespec time_start = {0, 0}, time_end = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time_start);
	for (int idx = 0; idx < 100; idx++) // 循环执行100次,测精准点
	{
		reduce<<<POW14, NTHREADS>>>(d_idata, d_odata, POW21); // 将2^21个输入计算出2^14个输出,存在d_odata中
		int s = POW14;
		while (s > 1)
		{
			cudaStatus = cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice); // 第一次循环,s=2^14 将d_odata的输出拷到d_intermediateSums
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMemcpy failed!");
				exit(1);
			}
			reduce<<<s / NTHREADS, NTHREADS>>>(d_intermediateSums, d_odata, s); // 第一次循环,s=2^14,将2^14个来自d_intermediateSums的输入,计算为2^7个输出,拷入d_odata中
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}
			s /= NTHREADS; // s /= 2^7, 经过一次循环问题规模已经缩小到原来的1/128
		}
		//  s = 1 时,最终的结果计算出来了,退出循环
		cudaStatus = cudaMemcpy(&res, d_odata, sizeof(int), cudaMemcpyDeviceToHost);// 将最终结果拷回cpu
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			exit(1);
		}
	}
	clock_gettime(CLOCK_REALTIME, &time_end);
	double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
	printf("duration:%.7lfms\n", costTime / 1000 / 1000);
	// 输出时间
	cudaFree(d_intermediateSums);
	return res;
}

__host__ int main()
{
	int *h_idata;
	h_idata = (int *)malloc(POW21BYTES);
	for (int i = 0; i < POW21; i++)
	{
		h_idata[i] = rand() % 256;
	}
	int cpu_result = 0;
	for (int i = 0; i < POW21; i++)
	{
		cpu_result += h_idata[i]; // cpu代码
	}
	int gpu_result = reduce_start(h_idata);
	printf("cpu_result: %d\n", cpu_result);
	printf("gpu_result: %d\n", gpu_result);
}
