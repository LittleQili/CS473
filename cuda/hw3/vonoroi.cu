#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <iterator>
#include <vector>
#include <algorithm>

#define NTHREADS 128
#define NUM_POINTS 500
#define LEN_LINE 1024
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

#pragma pack(1)

typedef struct
{
    short type; //0x4d42
    int size;   //width*
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct
{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;

typedef struct
{
    unsigned char r, g, b, alaph;
} mypoint;

void SaveBMPFile(mypoint *dst, unsigned int width, unsigned int height, const char *name)
{
    FILE *fd;
    fd = fopen(name, "wb");
    BMPHeader hdr;
    BMPInfoHeader InfoHdr;
    hdr.type = 0x4d42;
    hdr.size = width * height * 3 + sizeof(hdr) + sizeof(InfoHdr);
    hdr.reserved1 = 0;
    hdr.reserved2 = 0;
    hdr.offset = sizeof(hdr) + sizeof(InfoHdr);
    InfoHdr.size = sizeof(InfoHdr);
    InfoHdr.width = width;
    InfoHdr.height = height;
    InfoHdr.planes = 1;
    InfoHdr.bitsPerPixel = 24;
    InfoHdr.compression = 0;
    InfoHdr.imageSize = width * height * 3;
    InfoHdr.xPelsPerMeter = 0;
    InfoHdr.yPelsPerMeter = 0;
    InfoHdr.clrUsed = 0;
    InfoHdr.clrImportant = 0;
    fwrite(&hdr, sizeof(BMPHeader), 1, fd);
    fwrite(&InfoHdr, sizeof(BMPInfoHeader), 1, fd);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fputc(dst[y * width + x].b, fd);
            fputc(dst[y * width + x].g, fd);
            fputc(dst[y * width + x].r, fd);
        }
    }
    if (ferror(fd))
    {
        printf("***Unknown BMP load error.***\n");
        free(dst);
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("BMP file loaded successfully!\n");
    }
    fclose(fd);
}

// from ping to pong
__global__ void KerneljumpFlood(int SizeX, int SizeY, const float2 *SiteArray, int *Ping, int *Pong, int k, int *Mutex)
{
    //
    int pixelx = threadIdx.x + blockIdx.x * blockDim.x;
    int pixely = threadIdx.y + blockIdx.y * blockDim.y;
    int pixelIdx = pixelx + pixely * SizeX;

    int2 OffsetArray[8] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

    int seed = Ping[pixelIdx];
    if (seed < 0)
        return;

    for (int i = 0; i < 8; ++i)
    {
        int nextpixelx = pixelx + k * OffsetArray[i].x;
        int nextpixely = pixely + k * OffsetArray[i].y;
        if (nextpixelx >= 0 && nextpixelx < SizeX && nextpixely >= 0 && nextpixely < SizeY)
        {
            int nextpixelIdx = nextpixelx + nextpixely * SizeX;

            while (atomicCAS(Mutex, -1, nextpixelIdx) == nextpixelIdx)
            {
            }
            int nextseed = Pong[nextpixelIdx];

            if (nextseed < 0)
                Pong[nextpixelIdx] = seed;
            else
            {
                float2 P = make_float2(nextpixelx + 0.5f, nextpixely + 0.5f);

                float2 A = SiteArray[seed];
                float2 PA = make_float2(A.x - P.x, A.y - P.y);
                float PALength = PA.x * PA.x + PA.y * PA.y;

                float2 B = SiteArray[nextseed];
                float2 PB = make_float2(B.x - P.x, B.y - P.y);
                float PBLength = PB.x * PB.x + PB.y * PB.y;

                if (PALength < PBLength)
                    Pong[nextpixelIdx] = seed;
            }
            atomicExch(Mutex, -1);
        }
    }
}

__host__ void jumpFlood(int numPoints, int Size, std::vector<float2> &pointPos, std::vector<int> &seedVec, std::vector<uchar1> &colorLinear)
{
    size_t SiteSize = numPoints * sizeof(float2);
    float2 *SiteArray;
    size_t BufferSize = Size * Size * sizeof(int);
    int *Ping, *Pong;
    int *Mutex;
    CUDA_CALL(cudaMalloc(&SiteArray, SiteSize));
    CUDA_CALL(cudaMemcpy(SiteArray, &pointPos[0], SiteSize, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&Ping, BufferSize));
    CUDA_CALL(cudaMemcpy(Ping, &seedVec[0], BufferSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&Pong, BufferSize));
    CUDA_CALL(cudaMemcpy(Pong, Ping, BufferSize, cudaMemcpyDeviceToDevice));

    CUDA_CALL(cudaMalloc(&Mutex, sizeof(int)));
    CUDA_CALL(cudaMemset(Mutex, -1, sizeof(int)));

    cudaDeviceProp CudaDeviceProperty;
    cudaGetDeviceProperties(&CudaDeviceProperty, 0);
    dim3 BlockDim(CudaDeviceProperty.warpSize, CudaDeviceProperty.warpSize);
    dim3 GridDim((Size + BlockDim.x - 1) / BlockDim.x,
                 (Size + BlockDim.y - 1) / BlockDim.y);

    struct timespec time_start = {0, 0}, time_end = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_start);
    // struct timespec intime_start = {0, 0}, intime_end = {0, 0};
    // double incostTime = 0.0;
    for (int i = Size / 2; i > 0; i = i >> 1)
    {
        // clock_gettime(CLOCK_REALTIME, &intime_start);
        KerneljumpFlood<<<GridDim, BlockDim>>>(Size, Size, SiteArray, Ping, Pong, i, Mutex);
        cudaDeviceSynchronize();
        // clock_gettime(CLOCK_REALTIME, &intime_end);
        // incostTime += (intime_end.tv_sec - intime_start.tv_sec) * 1000 * 1000 * 1000 + intime_end.tv_nsec - intime_start.tv_nsec;

        CUDA_CALL(cudaMemcpy(Ping, Pong, BufferSize, cudaMemcpyDeviceToDevice));
        std::swap(Ping, Pong);
    }
    CUDA_CALL(cudaMemcpy(&seedVec[0], Pong, BufferSize, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &time_end);
    double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
    printf("JumpFlood cal cost:%.7lfms\n", costTime / 1000 / 1000);
    // printf("JumpFlood in cal cost:%.7lfms\n", incostTime / 1000 / 1000);

    CUDA_CALL(cudaFree(SiteArray));
    CUDA_CALL(cudaFree(Ping));
    CUDA_CALL(cudaFree(Pong));
    CUDA_CALL(cudaFree(Mutex));

    int sizeofimg = LEN_LINE * LEN_LINE * sizeof(mypoint);
    mypoint *img = (mypoint *)malloc(sizeofimg);
    assert(img);

    for (int y = 0; y < Size; ++y)
    {
        for (int x = 0; x < Size; ++x)
        {
            const int seed = seedVec[x + y * Size];
            if (seed != -1)
            {
                if (seed < 250)
                {
                    img[x + y * Size].g = (unsigned char)(seed + 1);
                    img[x + y * Size].b = 0;
                }
                else
                {
                    img[x + y * Size].g = 0;
                    img[x + y * Size].b = (unsigned char)(seed - 249);
                }
                img[x + y * Size].r = 0;
            }
        }
    }

    SaveBMPFile(img, LEN_LINE, LEN_LINE, "jumpflood.bmp");
    free(img);
}

__global__ void draw(mypoint *img, float *px, float *py, int numPoints, int Size)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int minpos = 0;
    float mind = (px[0] - x) * (px[0] - x) + (py[0] - y) * (py[0] - y);
    for (int i = 0; i < numPoints; ++i)
    {
        float distance = (px[i] - x) * (px[i] - x) + (py[i] - y) * (py[i] - y);
        if (distance < mind)
        {
            minpos = i;
            mind = distance;
        }
    }

    // extern __shared__ mypoint smem[]
    // smem[]
    img[y * Size + x].r = 0;
    if (minpos < 250)
    {
        img[y * Size + x].g = (unsigned char)(minpos + 1);
        img[x + y * Size].b = 0;
    }
    else
    {
        img[y * Size + x].g = 0;
        img[x + y * Size].b = (unsigned char)(minpos - 249);
    }
}

__host__ void drawline(int p1, int p2, std::vector<float2> &pointPos, mypoint *img, int size)
{
    int2 pointlow = (pointPos[p1].x < pointPos[p2].x ? make_int2((int)(pointPos[p1].x - 0.5), (int)(pointPos[p1].y - 0.5)) : make_int2((int)(pointPos[p2].x - 0.5), (int)(pointPos[p2].y - 0.5)));
    int2 pointhigh = (pointPos[p1].x >= pointPos[p2].x ? make_int2((int)(pointPos[p1].x - 0.5), (int)(pointPos[p1].y - 0.5)) : make_int2((int)(pointPos[p2].x - 0.5), (int)(pointPos[p2].y - 0.5)));
    // printf("low:(%d,%d),high(%d,%d) ", pointlow.x, pointlow.y, pointhigh.x, pointhigh.y);
    float div = (float)(pointhigh.y - pointlow.y) / (float)(pointhigh.x - pointlow.x);
    float offs = float(pointlow.y) - div * (float)(pointlow.x);
    if (div < 4.0 && div > -4.0)
    {
        for (int i = pointlow.x; i < pointhigh.x - 1; ++i)
        {
            int tmpy = int(float(i) * div + offs);
            img[tmpy * size + i].r = (unsigned char)(250);
            img[tmpy * size + i].g = (unsigned char)(250);
            img[tmpy * size + i].b = (unsigned char)(0);
            if (i + 1 < size)
            {
                img[tmpy * size + i + 1].r = (unsigned char)(250);
                img[tmpy * size + i + 1].g = (unsigned char)(250);
                img[tmpy * size + i + 1].b = (unsigned char)(0);
            }
            if ((tmpy + 1) < size)
            {
                img[(tmpy + 1) * size + i].r = (unsigned char)(250);
                img[(tmpy + 1) * size + i].g = (unsigned char)(250);
                img[(tmpy + 1) * size + i].b = (unsigned char)(0);
            }
        }
    }
    else
    {
        div = (float)(pointhigh.x - pointlow.x) / (float)(pointhigh.y - pointlow.y);
        offs = (float)(pointlow.x) - div * (float)(pointlow.y);
        if (pointhigh.y < pointlow.y)
            std::swap(pointhigh, pointlow);
        if (pointhigh.y < pointlow.y)
        {
            printf("ERROR IN DRAW LINE!\n");
            exit(0);
        }
        for (int i = pointlow.y; i < pointhigh.y - 1; ++i)
        {
            int tmpx = int(float(i) * div + offs);
            img[tmpx + i * size].r = (unsigned char)(250);
            img[tmpx + i * size].g = (unsigned char)(250);
            img[tmpx + i * size].b = (unsigned char)(0);
            if (tmpx + 1 < size)
            {
                img[tmpx + i * size + 1].r = (unsigned char)(250);
                img[tmpx + i * size + 1].g = (unsigned char)(250);
                img[tmpx + i * size + 1].b = (unsigned char)(0);
            }
            if (i + 1 < size)
            {
                img[tmpx + (i + 1) * size].r = (unsigned char)(250);
                img[tmpx + (i + 1) * size].g = (unsigned char)(250);
                img[tmpx + (i + 1) * size].b = (unsigned char)(0);
            }
            if (tmpx - 1 > 0)
            {
                img[tmpx + i * size - 1].r = (unsigned char)(250);
                img[tmpx + i * size - 1].g = (unsigned char)(250);
                img[tmpx + i * size - 1].b = (unsigned char)(0);
            }
            if (i - 1 > 0)
            {
                img[tmpx + (i - 1) * size].r = (unsigned char)(250);
                img[tmpx + (i - 1) * size].g = (unsigned char)(250);
                img[tmpx + (i - 1) * size].b = (unsigned char)(0);
            }
        }
    }
}

__host__ void drawTriangle(mypoint *img, int numPoints, int Size, std::vector<float2> &pointPos)
{
    int pairedpoint[numPoints][numPoints] = {{0}};
    for (int x = 0; x < Size - 1; ++x)
    {
        for (int y = 0; y < Size - 1; ++y)
        {
            int pair1, pair2;
            if (img[y * Size + x].g != img[y * Size + x + 1].g || img[y * Size + x].b != img[y * Size + x + 1].b)
            {
                if (img[y * Size + x].g == 0)
                    pair1 = (int)(img[y * Size + x].b + 249);
                else
                    pair1 = (int)(img[y * Size + x].g - 1);
                if (img[y * Size + x + 1].g == 0)
                    pair2 = (int)(img[y * Size + x + 1].b + 249);
                else
                    pair2 = (int)(img[y * Size + x + 1].g - 1);
                pairedpoint[pair1][pair2] = 1;
                pairedpoint[pair2][pair1] = 1;
            }
            if (img[y * Size + x].g != img[(y + 1) * Size + x].g || img[y * Size + x].b != img[(y + 1) * Size + x].b)
            {
                if (img[y * Size + x].g == 0)
                    pair1 = (int)(img[y * Size + x].b + 249);
                else
                    pair1 = (int)(img[y * Size + x].g - 1);
                if (img[(y + 1) * Size + x].g == 0)
                    pair2 = (int)(img[(y + 1) * Size + x].b + 249);
                else
                    pair2 = (int)(img[(y + 1) * Size + x].g - 1);
                pairedpoint[pair1][pair2] = 1;
                pairedpoint[pair2][pair1] = 1;
            }
        }
    }
    for (int i = 0; i < numPoints - 1; ++i)
    {
        for (int j = i + 1; j < numPoints; ++j)
        {
            // if(pairedpoint[i][j] == 1)printf("(%d,%d) ",i,j);
            if (pairedpoint[i][j] == 1)
                drawline(i, j, pointPos, img, Size);
        }
        // printf("\n");
    }
}

__host__ void naive(int numPoints, int Size, std::vector<float2> &pointPos)
{
    int sizeofimg = Size * Size * sizeof(mypoint);
    int sizeofpoints = sizeof(float) * numPoints;
    mypoint *img = (mypoint *)malloc(sizeofimg);
    assert(img);
    mypoint *cudaimg;
    CUDA_CALL(cudaMalloc((void **)&cudaimg, sizeofimg));

    float tmppointx[numPoints];
    float tmppointy[numPoints];
    for (int i = 0; i < numPoints; ++i)
    {
        tmppointx[i] = pointPos[i].x;
        tmppointy[i] = pointPos[i].y;
    }
    float *pointX, *pointY;
    CUDA_CALL(cudaMalloc((void **)&pointX, sizeofpoints));
    CUDA_CALL(cudaMalloc((void **)&pointY, sizeofpoints));
    CUDA_CALL(cudaMemcpy(pointX, &tmppointx[0], sizeofpoints, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pointY, &tmppointy[0], sizeofpoints, cudaMemcpyHostToDevice));

    struct timespec time_start = {0, 0}, time_end = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_start);

    draw<<<Size, Size>>>(cudaimg, pointX, pointY, numPoints, Size);
    CUDA_CALL(cudaMemcpy(img, cudaimg, sizeofimg, cudaMemcpyDeviceToHost));
    // for(int i = 0;i < width * height;++i){
    //     printf("r%d g%d b%d, ",img[i].r,img[i].g,img[i].b);
    // }
    clock_gettime(CLOCK_REALTIME, &time_end);
    double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
    printf("Naive cal cost:%.7lfms\n", costTime / 1000 / 1000);
    drawTriangle(img, numPoints, Size, pointPos);
    SaveBMPFile(img, LEN_LINE, LEN_LINE, "naive.bmp");

    CUDA_CALL(cudaFree(cudaimg));
    CUDA_CALL(cudaFree(pointX));
    CUDA_CALL(cudaFree(pointY));

    free(img);
}

__host__ int main()
{
    int numPoints = NUM_POINTS;
    int Size = LEN_LINE;

    std::vector<float2> pointPos;
    std::vector<int> seedVec1(Size * Size, -1);
    std::vector<int> seedVec2(Size * Size, -1);
    std::vector<uchar1> colorLinear;
    srand(time(NULL));
    for (int i = 0; i < numPoints; ++i)
    {
        float X = static_cast<float>(rand()) / RAND_MAX * Size;
        float Y = static_cast<float>(rand()) / RAND_MAX * Size;
        int pixelx = static_cast<int>(floorf(X));
        int pixely = static_cast<int>(floorf(Y));

        pointPos.push_back(make_float2(pixelx + 0.5f, pixely + 0.5f));
        seedVec1[pixelx + pixely * Size] = i;
        seedVec2[pixelx + pixely * Size] = i;

        colorLinear.push_back(make_uchar1(static_cast<unsigned char>(static_cast<float>(i) / numPoints * 255.0f)));
    }

    jumpFlood(numPoints, Size, pointPos, seedVec1, colorLinear);
    naive(numPoints, Size, pointPos);
    return 0;
}