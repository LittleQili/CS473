#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <iterator>
#include <vector>

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
    struct timespec intime_start = {0, 0}, intime_end = {0, 0};
    double incostTime = 0.0;
    for (int i = Size / 2; i > 0; i = i >> 1)
    {
        clock_gettime(CLOCK_REALTIME, &intime_start);
        KerneljumpFlood<<<GridDim, BlockDim>>>(Size, Size, SiteArray, Ping, Pong, i, Mutex);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &intime_end);
        incostTime += (intime_end.tv_sec - intime_start.tv_sec) * 1000 * 1000 * 1000 + intime_end.tv_nsec - intime_start.tv_nsec;

        CUDA_CALL(cudaMemcpy(Ping, Pong, BufferSize, cudaMemcpyDeviceToDevice));
        std::swap(Ping, Pong);
    }
    CUDA_CALL(cudaMemcpy(&seedVec[0], Pong, BufferSize, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &time_end);
    double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
    printf("JumpFlood cal cost:%.7lfms\n", costTime / 1000 / 1000);
    printf("JumpFlood in cal cost:%.7lfms\n", incostTime / 1000 / 1000);

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
                    img[x + y * Size].g = (unsigned char)(seed);
                    img[x + y * Size].b = (unsigned char)(seed);
                }
                else
                {
                    img[x + y * Size].g = (unsigned char)(500 - seed);
                    img[x + y * Size].b = (unsigned char)(seed - 250);
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
        img[y * Size + x].g = (unsigned char)(minpos);
        img[x + y * Size].b = (unsigned char)(minpos);
    }
    else
    {
        img[y * Size + x].g = (unsigned char)(500 - minpos);
        img[x + y * Size].b = (unsigned char)(minpos - 250);
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