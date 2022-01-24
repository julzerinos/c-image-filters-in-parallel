#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

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


    int is_filter_functional;
    void (*filter_function)(int *, int *, int *);

    switch (filterType) {
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
        default:
            puts("wrong filter type parameter");
            exit(-1);
            break;
    }
    struct timeval startMpi, endMpi;
    gettimeofday(&startMpi, 0);
    //MPI
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int heightPerWatek = height / numprocs;

    if(heightPerWatek * numprocs != height)
        error("wrong image format");



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

    RGBTRIPLE(*localdata)[width] = calloc(height / numprocs, width * sizeof(RGBTRIPLE));
    int danePerWatek = height * width / numprocs;
    RGBTRIPLE(*obrazekkoncowy)[width] = calloc(height, width * sizeof(RGBTRIPLE));


//    mpirun -np 2 ./aaa -f -0 milla.bmp dadd.bmp
//    mpicc -o aaa -g mainmpi.c filters/convolution.c filters/functional.c benchmarking/benchmark.c -lm -fopenmp -lmpi




    MPI_Scatter(imageparalel, danePerWatek, rgb_triple, localdata,
                danePerWatek, rgb_triple, 0, MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    apply_functional_sequentiallyMPI(heightPerWatek, width, 0, heightPerWatek, localdata, filter_function);

    MPI_Gather(localdata, danePerWatek, rgb_triple, obrazekkoncowy, danePerWatek, rgb_triple, 0, MPI_COMM_WORLD);




    gettimeofday(&endMpi, 0);
    double timeMpi = get_elapsed_time(startMpi, endMpi);


    if (rank == 0) {
        fprintf(stdout, "[log] algorithm completed\n  1. sequential timing (micro seconds): %.3f\n", timeMpi * 1000);

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