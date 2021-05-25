#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct
{
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;

#define BLOCK_SIZE 16

// get element, within a block
// row, col: threadIdx
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// set element, within a block
// row, col: threadIdx
__device__ void SetElement(const Matrix A, int row, int col, float val)
{
    A.elements[row * A.stride + col] = val;
}
// read one block
// row & col: blockIdx
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    // here, row&col is for C.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    int row = threadIdx.y;
    int col = threadIdx.x;
    float CValue = 0;
    // 外层循环：为了计算出一个subC矩阵，需要读多个A B block.
    // A.width是两个矩阵乘所相同的维度
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
    {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // 每个线程分别去做自己的global memory访问
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        // 每个线程计算读到shared memory部分的乘法部分
        for (int e = 0; e < BLOCK_SIZE; ++e)
        {
            CValue += As[row][e] * Bs[e][col];
        }

        // 这个同步不是很理解？???
        // 或许是为下一次访存做准备？
        __syncthreads();
    }

    // 写回subC
    SetElement(Csub, row, col, CValue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // load A B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = A.width;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = B.width;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // allocate C
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    d_C.stride = C.width;
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
    int awidth = 64, aheight = 64, bwidth = 64;
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
        A.elements[i] = (rand()) % 100;
        printf("%.1f ", A.elements[i]);
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < B.width * B.height; ++i)
    {
        B.elements[i] = (rand()) % 100;
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