#include <cuda.h>
#include <stdio.h>


__global__ void print()
{
    unsigned int tid = threadIdx.x;
    unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalTid < 128)
        printf("Hello from %d %d\n", tid, globalTid);
}

int main()
{
    cudaError_t cudaStatus;
    print<<<1, 1025>>>();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "printKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(0);
    }
    cudaDeviceSynchronize();
}
