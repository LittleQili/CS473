#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct
{
    int width;
    int height;
    float *elements;
} Matrix;

#define BLOCK_SIZE 16

// 这里的kernel是，单个kernel读两次global memory，写一次global memory，每次计算出来一个值。
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float CValue = 0;
    for (int e = 0; e < A.width; ++e)
    {
        CValue += A.elements[row * A.width + e] * B.elements[e * B.height + col];
    }
    
    C.elements[row * C.width + col] = CValue;
}

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // load A B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // allocate C
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // printf("Matrix A:\n");
    // for (int i = 0; i < A.width * A.height; ++i)
    // {
    //     printf("%.1f ", A.elements[i]);
    // }
    // printf("\nMatrix B:\n");
    // for (int i = 0; i < B.width * B.height; ++i)
    // {
    //     printf("%.1f ", B.elements[i]);
    // }

    // Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // printf("\nMatrix C:\n");
    // for (int i = 0; i < C.width * C.height; ++i)
    // {
    //     printf("%.1f ", C.elements[i]);
    // }

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main()
{
    int awidth = 64, aheight = 32, bwidth = 64;
    Matrix A, B;
    A.width = awidth;
    A.height = aheight;
    B.width = bwidth;
    B.height = awidth;
    int asize = A.width * A.height * sizeof(float);
    int bsize = B.width * B.height * sizeof(float);
    A.elements = (float *)malloc(asize);
    B.elements = (float *)malloc(bsize);

    srand((unsigned)time(NULL));
    printf("Matrix A:\n");
    for (int i = 0; i < A.width * A.height; ++i)
    {
        A.elements[i] = (rand())%100;
        printf("%.1f ", A.elements[i]);
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < B.width * B.height; ++i)
    {
        B.elements[i] = (rand())%100;
        printf("%.1f ", B.elements[i]);
    }

    Matrix C;
    C.width = B.width;
    C.height = A.height;
    int csize = C.width * C.height * sizeof(float);
    C.elements = (float *)malloc(csize);

    MatMul(A, B, C);

    printf("\nMatrix C:\n");
    for (int i = 0; i < C.width * C.height; ++i)
    {
        printf("%.1f ", C.elements[i]);
    }

    free(A.elements);
    free(B.elements);
    free(C.elements);
}