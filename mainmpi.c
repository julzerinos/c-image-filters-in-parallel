#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <mpi.h>

#include "filters/convolution.h"
#include "filters/functional.h"
#include "benchmarking/benchmark.h"
#include "bmp.h"

void error(char *message) {
    fprintf(stderr, "[error] %s\n", message);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    struct stat filestat;
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);


    char filterType = getopt(argc, argv, "cf");
    if (filterType == '?')
        error("no filterType detected in options");

    char filterNo = getopt(argc, argv, "01");
    if (filterNo == '?')
        error("no filterNo detected in options");

    char *infile = argv[optind];
    char *outfile = argv[optind + 1];

    FILE *inptr = fopen(infile, "r");
    if (inptr == NULL)
        error("failed to open input file");

    BITMAPFILEHEADER bf;
    BITMAPINFOHEADER bi;
    fread(&bf, sizeof(BITMAPFILEHEADER), 1, inptr);
    fread(&bi, sizeof(BITMAPINFOHEADER), 1, inptr);
    if (bf.bfType != 0x4d42 || bf.bfOffBits != 54 || bi.biSize != 40 || bi.biBitCount != 24 || bi.biCompression != 0) {
        fclose(inptr);
        error("input file has incorrect format (not 24-bit bmp)");
    }


    int height = abs(bi.biHeight);
    int width = bi.biWidth;
    RGBTRIPLE(*image)[width] = calloc(height, width * sizeof(RGBTRIPLE));
    if (image == NULL) {
        fclose(inptr);
        error("memory allocation for image failed");
    }

    int padding = (4 - (width * sizeof(RGBTRIPLE)) % 4) % 4;
    for (int i = 0; i < height; i++) {
        fread(image[i], sizeof(RGBTRIPLE), width, inptr);
        fseek(inptr, padding, SEEK_CUR);
    }
    fclose(inptr);

    RGBTRIPLE(*imageparalel)[width] = calloc(height, width * sizeof(RGBTRIPLE));
    memcpy(imageparalel, image, height * width * sizeof(RGBTRIPLE));


    int kernel_dimension = 3;
    if (kernel_dimension % 2 == 0)
        error("kernel dimension must be an odd number");
    double kernel[kernel_dimension][kernel_dimension];


    int is_filter_functional;
    void (*filter_function)(int *, int *, int *);

    void (*apply_convolutional_sequentially)();
    void (*apply_convolutional_parallelly)();

    switch (filterType) {
        case 'c':
            switch (filterNo) {
                case '0':
                    apply_convolutional_sequentially = &blur_sequentially;
                    apply_convolutional_parallelly = &blur_parallelly;
                    break;
                case '1':
                    apply_convolutional_sequentially = &edge_detection_sequentially;
                    apply_convolutional_parallelly = &edge_detection_parallelly;
                    break;
            }
            is_filter_functional = 0;
            break;

        case 'f':
            switch (filterNo) {
                case '0':
                    filter_function = &grayscale;
                    break;
                case '1':
                    filter_function = &inversion;
                    break;
            }
            is_filter_functional = 1;
            break;
    }
//    // SEQUENTIALLY
    struct timeval startSeq, endSeq;
    gettimeofday(&startSeq, 0);
//
//
//    if (is_filter_functional)
//        apply_functional_sequentially(height, width, image, filter_function);
//
//    else
//        apply_convolutional_sequentially(height, width, image, kernel_dimension, kernel);
//

//    int a=1,b=2,c=3;
//
//    filter_function(&a,&b,&c);
//    printf("AAAAAAAAAAAAAAAAAAAasfasfABBBBBBBBrgdgdrBBBBBBBBBBB");
//    puts("aegeg");


    gettimeofday(&endSeq, 0);
    double timeSeq = get_elapsed_time(startSeq, endSeq);
//
//
//
//
//    // THREADS
    struct timeval startThrd, endThrd;
//    gettimeofday(&startThrd, 0);
//
//
//    if (is_filter_functional)
//        apply_functional_parallelly(1, height, width, imageparalel, filter_function);
//    else
//        apply_convolutional_parallelly(20, height, width, imageparalel, kernel_dimension, kernel);
//
//
    gettimeofday(&endThrd, 0);
    double timeThrd = get_elapsed_time(startThrd, endThrd);


    //MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int iloscwatkow = 4; //todo AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    int *globaldata = NULL;
    RGBTRIPLE(*localdata)[width] = calloc(height, width * sizeof(RGBTRIPLE));
//    memcpy(imageparalel, image, height*width * sizeof(RGBTRIPLE));
//    memcpy(imageparalel, image, height*width * sizeof(RGBTRIPLE));
//    int size = 20;

//    if (rank == 0) {
//        globaldata = malloc(size * sizeof(int) );
//        for (int i=0; i<size; i++)
//            globaldata[i] = 2*i+1;
//
//        printf("Processor %d has data: ", rank);
//        for (int i=0; i<size; i++)
//            printf("%d ", globaldata[i]);
//        printf("\n");
//    }

    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_BYTE, MPI_BYTE, MPI_BYTE};
    MPI_Datatype rgb_triple;
    MPI_Aint offsets[3];

    offsets[0] = offsetof(RGBTRIPLE, rgbtBlue);
    offsets[1] = offsetof(RGBTRIPLE, rgbtGreen);
    offsets[2] = offsetof(RGBTRIPLE, rgbtRed);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &rgb_triple);
    MPI_Type_commit(&rgb_triple);


//    rgb_triple(*  )[width] = calloc(height, width * sizeof(RGBTRIPLE));
//    mpicc -o aaa -g mainmpi.c filters/convolution.c filters/functional.c benchmarking/benchmark.c -lm -fopenmp -lmpi

    int a = height * width;
    printf("AAAA %d", a);
    puts("A");
int bb = 44100;
    printf("%d, %d, %p, %p, %p", height, width, localdata, &imageparalel, &localdata);
    printf("%d, %d, %d", imageparalel[0][0].rgbtRed, imageparalel[0][0].rgbtGreen, imageparalel[0][0].rgbtBlue);


    puts("AAAAAAAA");

    MPI_Scatter(imageparalel, bb, rgb_triple, localdata,
                bb, rgb_triple, 0, MPI_COMM_WORLD);

    for(int i =0; i < height;++i)
        printf("%d, %d, %d \n", localdata[0][i].rgbtRed, localdata[0][i].rgbtGreen, localdata[0][i].rgbtBlue);
//
//    for (int i = 0; i < height; i++)
//        for (int j = 0; j < width; j++) {
//            apply_functional(i, j, height, width, image, filter_function);
//        }

    apply_functional_sequentially(height, width, localdata, filter_function);
    puts("AAAAAAAA");
    puts("AAAAAAAA");
    puts("AAAAAAAA");

//    printf("Processor %d doubling the data, now has %d\n", rank, localdata);

    MPI_Gather(localdata, bb, rgb_triple, imageparalel, bb, rgb_triple, 0, MPI_COMM_WORLD);
    puts("DDDD");
    puts("AAADDDDAAAAA");
//    if (rank == 0) {
//        printf("Processor %d has data: ", rank);
//        for (int i=0; i<size; i++)
//            printf("%d ", globaldata[i]);
//        printf("\n");
//    }
//
//    if (rank == 0)
//        free(globaldata);

    MPI_Finalize();


    fprintf(stdout, "[log] algorithm completed\n  1. sequential timing (micro seconds): %.3f\n", timeSeq * 1000);
    fprintf(stdout, "[log] algorithm completed\n  1. thread timing (micro seconds): %.3f\n", timeThrd * 1000);


    FILE *outptr = fopen(outfile, "w");
    if (outptr == NULL)
        error("could not create/open output file");

    fwrite(&bf, sizeof(BITMAPFILEHEADER), 1, outptr);
    fwrite(&bi, sizeof(BITMAPINFOHEADER), 1, outptr);

    for (int i = 0; i < height; i++) {
        fwrite(image[i], sizeof(RGBTRIPLE), width, outptr);
        for (int k = 0; k < padding; k++)
            fputc(0x00, outptr);
    }

    free(image);
    fclose(outptr);


    FILE *paraout = fopen("parallel.bmp", "w");
    if (paraout == NULL)
        error("could not create/open output file");

    fwrite(&bf, sizeof(BITMAPFILEHEADER), 1, paraout);
    fwrite(&bi, sizeof(BITMAPINFOHEADER), 1, paraout);

    for (int i = 0; i < height; i++) {
        fwrite(imageparalel[i], sizeof(RGBTRIPLE), width, paraout);
        for (int k = 0; k < padding; k++)
            fputc(0x00, paraout);
    }

    free(imageparalel);
    fclose(paraout);

//    MPI_Finalize();

    return 0;
}