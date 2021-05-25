#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// #include <conio.h>

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int)*ARRAY_SIZE) 
#define CUDA_CALL(x) {const cudaError_t a = (x); if(a != cudaSuccess){ printf("\nCUDA Error: %s (err_num = %d)\n", cudaGetErrorString(a),a); cudaDeviceReset(); assert(0);}}

__host__ void cuda_error_check(const char *prefix, const char *postfix){
    if (cudaPeekAtLastError()!= cudaSuccess){
        printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        cudaDeviceReset();
        // wait_exit();
        exit(1);
	}
}

__global__ void kernel_id(unsigned int * const block,
                          unsigned int * const thread,
                          unsigned int * const warp,
                          unsigned int * const calc_therad){
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_therad[thread_idx] = thread_idx;
}



int main(){
    unsigned int cpu_block[ARRAY_SIZE];
    unsigned int cpu_thread[ARRAY_SIZE];
    unsigned int cpu_warp[ARRAY_SIZE];
    unsigned int cpu_calc_thread[ARRAY_SIZE];

    const unsigned int num_blocks = 2;
    const unsigned int num_threadpblock = ARRAY_SIZE/num_blocks;
    // char ch;

    unsigned int *gpu_block;
    unsigned int *gpu_thread;
    unsigned int *gpu_warp;
    unsigned int *gpu_calc_thread;

    unsigned int i;

    CUDA_CALL(cudaMalloc((void**)&gpu_block,ARRAY_SIZE_IN_BYTES));
    CUDA_CALL(cudaMalloc((void**)&gpu_thread,ARRAY_SIZE_IN_BYTES));
    CUDA_CALL(cudaMalloc((void**)&gpu_warp,ARRAY_SIZE_IN_BYTES));
    CUDA_CALL(cudaMalloc((void**)&gpu_calc_thread,ARRAY_SIZE_IN_BYTES));

    kernel_id<<<num_blocks,num_threadpblock>>>(gpu_block,gpu_thread,gpu_warp,gpu_calc_thread);
    cuda_error_check("Error ", " returned from kernel_id kernel");

    CUDA_CALL(cudaMemcpy(cpu_block,gpu_block,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cpu_thread,gpu_thread,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cpu_warp,gpu_warp,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cpu_calc_thread,gpu_calc_thread,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(gpu_block));
    CUDA_CALL(cudaFree(gpu_thread));
    CUDA_CALL(cudaFree(gpu_warp));
    CUDA_CALL(cudaFree(gpu_calc_thread));

    for(i = 0;i < ARRAY_SIZE;i++){
        printf("Calculated Thread: %3u - Block : %2u - Warp : %2u - Thread : %3u\n",
            cpu_calc_thread[i],cpu_block[i],cpu_warp[i],cpu_thread[i]);
    }
    // ch = getch();
}