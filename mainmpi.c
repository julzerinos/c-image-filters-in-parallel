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

    struct stat filestat;
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

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



//    mpirun -np 2 ./aaa -f -0 milla.bmp dadd.bmp
//    mpicc -o aaa -g mainmpi.c filters/convolution.c filters/functional.c benchmarking/benchmark.c -lm -fopenmp -lmpi

    int watki = 4;
    RGBTRIPLE(*localdata)[width] = calloc(height / watki, width * sizeof(RGBTRIPLE));
    int heightPerWatek = height / watki;
    int danePerWatek = height * width / watki;
    RGBTRIPLE(*obrazekkoncowy)[width] = calloc(height, width * sizeof(RGBTRIPLE));

    MPI_Scatter(imageparalel, danePerWatek, rgb_triple, localdata,
                danePerWatek, rgb_triple, 0, MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    apply_functional_sequentiallyMPI(heightPerWatek, width, 0, heightPerWatek, localdata, filter_function);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(localdata, danePerWatek, rgb_triple, obrazekkoncowy, danePerWatek, rgb_triple, 0, MPI_COMM_WORLD);

    if (rank == 0) {
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
            fwrite(obrazekkoncowy[i], sizeof(RGBTRIPLE), width, paraout);
            for (int k = 0; k < padding; k++)
                fputc(0x00, paraout);
        }

        free(imageparalel);
        free(obrazekkoncowy);
        fclose(paraout);
    }
    MPI_Finalize();

    return 0;
}