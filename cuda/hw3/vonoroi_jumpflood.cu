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

__global__ void Kernel( int SizeX , int SizeY , const float2 * SiteArray , const int * Ping , int * Pong , int k , int * Mutex )
{
    //
    const int CellX = threadIdx.x + blockIdx.x * blockDim.x ;
    const int CellY = threadIdx.y + blockIdx.y * blockDim.y ;

    const int CellIdx = CellX + CellY * SizeX ;
    const int Seed = Ping[CellIdx] ;
    if ( Seed < 0 )
    {
        return ;
    }

    //
    const int2 OffsetArray[8] = { { - 1 , - 1 } ,
                                  {   0 , - 1 } ,
                                  {   1 , - 1 } ,
                                  { - 1 ,   0 } ,
                                  {   1 ,   0 } ,
                                  { - 1 ,   1 } ,
                                  {   0 ,   1 } ,
                                  {   1 ,   1 } } ;

    for ( int i = 0 ; i < 8 ; ++ i )
    {
        const int FillCellX = CellX + k * OffsetArray[i].x ;
        const int FillCellY = CellY + k * OffsetArray[i].y ; 
        if ( FillCellX >= 0 && FillCellX < SizeX && FillCellY >= 0 && FillCellY < SizeY )
        {
            //
            const int FillCellIdx = FillCellX + FillCellY * SizeX ;

            // Lock
            //
            while ( atomicCAS( Mutex , - 1 , FillCellIdx ) == FillCellIdx )
            {
            }

            const int FillSeed = Pong[FillCellIdx] ;

            if ( FillSeed < 0 )
            {
                Pong[FillCellIdx] = Seed ;
            }
            else
            {
                float2 P = make_float2( FillCellX + 0.5f , FillCellY + 0.5f ) ;

                float2 A = SiteArray[Seed] ;
                float2 PA = make_float2( A.x - P.x , A.y - P.y ) ;
                float PALength = PA.x * PA.x + PA.y * PA.y ;

                const float2 B = SiteArray[FillSeed] ;
                float2 PB = make_float2( B.x - P.x , B.y - P.y ) ;
                float PBLength = PB.x * PB.x + PB.y * PB.y ;

                if ( PALength < PBLength )
                {
                    Pong[FillCellIdx] = Seed ;
                }
            }

            // Release
            //
            atomicExch( Mutex , - 1 ) ;
        }
    }
}

__host__ int main()
{
    //
    int NumSites = NUM_POINTS ;
    int Size     = LEN_LINE ;

    //
    int NumCudaDevice = 0 ;
    cudaGetDeviceCount( & NumCudaDevice ) ;
    if ( ! NumCudaDevice )
    {
        return EXIT_FAILURE ;
    }

    //
    //
    std::vector< float2 > SiteVec ;
    std::vector< int >    SeedVec( Size * Size , - 1 ) ;
    std::vector< uchar3 > RandomColorVec ;
    for ( int i = 0 ; i < NumSites ; ++ i )
    {
        float X = static_cast< float >( rand() ) / RAND_MAX * Size ;
        float Y = static_cast< float >( rand() ) / RAND_MAX * Size ;
        int CellX = static_cast< int >( floorf( X ) ) ;
        int CellY = static_cast< int >( floorf( Y ) ) ;

        SiteVec.push_back( make_float2( CellX + 0.5f , CellY + 0.5f ) ) ;
        SeedVec[CellX + CellY * Size] = i ;

        RandomColorVec.push_back( make_uchar3( static_cast< unsigned char >( static_cast< float >( i ) / NumSites * 255.0f ) ,
                                               static_cast< unsigned char >( static_cast< float >( i ) / NumSites * 255.0f ) ,
                                               static_cast< unsigned char >( static_cast< float >( i ) / NumSites * 255.0f ) ) ) ;
    }

    //
    size_t SiteSize = NumSites * sizeof( float2 ) ;

    float2 * SiteArray = NULL ;
    cudaMalloc( & SiteArray , SiteSize ) ;
    cudaMemcpy( SiteArray , & SiteVec[0] , SiteSize , cudaMemcpyHostToDevice ) ;

    //
    size_t BufferSize = Size * Size * sizeof( int ) ;

    int * Ping = NULL , * Pong = NULL ;
    cudaMalloc( & Ping , BufferSize ) , cudaMemcpy( Ping , & SeedVec[0] , BufferSize , cudaMemcpyHostToDevice ) ;
    cudaMalloc( & Pong , BufferSize ) , cudaMemcpy( Pong , Ping , BufferSize , cudaMemcpyDeviceToDevice ) ;

    //
    int * Mutex = NULL ;
    cudaMalloc( & Mutex , sizeof( int ) ) , cudaMemset( Mutex , - 1 , sizeof( int ) ) ;

    //
    //
    cudaDeviceProp CudaDeviceProperty ;
    cudaGetDeviceProperties( & CudaDeviceProperty , 0 ) ;

    dim3 BlockDim( CudaDeviceProperty.warpSize , CudaDeviceProperty.warpSize ) ;
    dim3 GridDim( ( Size + BlockDim.x - 1 ) / BlockDim.x ,
                  ( Size + BlockDim.y - 1 ) / BlockDim.y ) ;

    for ( int k = Size / 2 ; k > 0 ; k = k >> 1 )
    {
        Kernel<<< GridDim , BlockDim >>>( Size , Size , SiteArray , Ping , Pong , k , Mutex ) ;
        cudaDeviceSynchronize() ;

        cudaMemcpy( Ping , Pong , BufferSize , cudaMemcpyDeviceToDevice ) ;
        std::swap( Ping , Pong ) ;
    }
    cudaMemcpy( & SeedVec[0] , Pong , BufferSize , cudaMemcpyDeviceToHost ) ;

    //
    cudaFree( SiteArray ) ;
    cudaFree( Ping ) ;
    cudaFree( Pong ) ;
    cudaFree( Mutex ) ;

    //
    //
    int sizeofimg = LEN_LINE * LEN_LINE * sizeof(mypoint);
    mypoint *img = (mypoint *)malloc(sizeofimg);
    assert(img);

    // std::vector< uchar3 > Pixels( Size * Size ) ;
    for ( int y = 0 ; y < Size ; ++ y )
    {
        for ( int x = 0 ; x < Size ; ++ x )
        {
            const int Seed = SeedVec[x + y * Size] ;
            if ( Seed != - 1 )
            {
                img[x + y * Size].r = 0 ;
                img[x + y * Size].g = RandomColorVec[Seed].y ;
                img[x + y * Size].b = 0 ;
            }
        }
    }

    SaveBMPFile(img, LEN_LINE, LEN_LINE, "gpu.bmp");
    free(img);

    return EXIT_SUCCESS ;
}