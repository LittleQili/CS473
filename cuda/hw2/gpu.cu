#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define NTHREADS 120

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

__global__ void draw(mypoint *img)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    int r = (int)((x / 640.0-y / 1105.5125) * 256.0);
    int g = (int)(y / 2.165);
    int b = 256 - r - g;
    if (r < 0 || g < 0 || b < 0)
        r = g = b = 0;

    // extern __shared__ mypoint smem[]
    // smem[]
    img[y * 640 + x].r = r;
    img[y * 640 + x].g = g;
    img[y * 640 + x].b = b;
}

__host__ int main()
{
    int width = 640, height = 640;
    int sizeofimg = width * height * sizeof(mypoint);
    mypoint *img = (mypoint *)malloc(sizeofimg);
    assert(img);
    mypoint *cudaimg;
    CUDA_CALL(cudaMalloc((void **)&cudaimg, sizeofimg));

    struct timespec time_start = {0, 0}, time_end = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_start);

    draw<<<width,height>>>(cudaimg);
    CUDA_CALL(cudaMemcpy(img, cudaimg, sizeofimg, cudaMemcpyDeviceToHost));
    // for(int i = 0;i < width * height;++i){
    //     printf("r%d g%d b%d, ",img[i].r,img[i].g,img[i].b);
    // }
    clock_gettime(CLOCK_REALTIME, &time_end);
    double costTime = (time_end.tv_sec - time_start.tv_sec) * 1000 * 1000 * 1000 + time_end.tv_nsec - time_start.tv_nsec;
    printf("GPU cal cost:%.7lfms\n", costTime / 1000 / 1000);
    SaveBMPFile(img, width, height, "gpu.bmp");

    CUDA_CALL(cudaFree(cudaimg));

    free(img);
}