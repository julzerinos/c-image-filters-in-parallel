#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>

#include "filters/convolution.h"
#include "filters/functional.h"
#include "benchmarking/benchmark.h"
#include "bmp.h"

void error(char *message)
{
    fprintf(stderr, "[error] %s\n", message);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    char filter = getopt(argc, argv, "gb");
    if (filter == '?')
        error("no filter detected in options");

    char *infile = argv[optind];
    char *outfile = argv[optind + 1];

    FILE *inptr = fopen(infile, "r");
    if (inptr == NULL)
        error("failed to open input file");

    BITMAPFILEHEADER bf;
    BITMAPINFOHEADER bi;
    fread(&bf, sizeof(BITMAPFILEHEADER), 1, inptr);
    fread(&bi, sizeof(BITMAPINFOHEADER), 1, inptr);
    if (bf.bfType != 0x4d42 || bf.bfOffBits != 54 || bi.biSize != 40 || bi.biBitCount != 24 || bi.biCompression != 0)
    {
        fclose(inptr);
        error("input file has incorrect format (not 24-bit bmp)");
    }

    int height = abs(bi.biHeight);
    int width = bi.biWidth;
    RGBTRIPLE(*image)
    [width] = calloc(height, width * sizeof(RGBTRIPLE));
    if (image == NULL)
    {
        fclose(inptr);
        error("memory allocation for image failed");
    }

    int padding = (4 - (width * sizeof(RGBTRIPLE)) % 4) % 4;
    for (int i = 0; i < height; i++)
    {
        fread(image[i], sizeof(RGBTRIPLE), width, inptr);
        fseek(inptr, padding, SEEK_CUR);
    }
    fclose(inptr);

    int kernel_dimension = 5;
    if (kernel_dimension % 2 == 0)
        error("kernel dimension must be an odd number");
    double kernel[kernel_dimension][kernel_dimension];

    int is_filter_functional;
    void (*filter_function)(int *, int *, int *);

    switch (filter)
    {
    case 'b':
        create_blur_kernel(kernel_dimension, kernel);
        is_filter_functional = 0;
        break;

    case 'g':
        filter_function = &grayscale;
        is_filter_functional = 1;
        break;
    }

    // SEQUENTIALLY

    clock_t end, start = clock();

    if (is_filter_functional)
        apply_functional_sequentially(height, width, image, filter_function);
    else
        apply_convolution_sequentially(height, width, image, kernel_dimension, kernel);

    end = clock();

    double sequential_clock_time = benchmark_time(start, end);

    // THREADS
    // todo

    // MPI
    // todo

    // CUDA
    // todo

    fprintf(stdout, "[log] algorithm completed\n  1. sequential timing (clock ticks): %f\n", sequential_clock_time);

    FILE *outptr = fopen(outfile, "w");
    if (outptr == NULL)
        error("could not create/open output file");

    fwrite(&bf, sizeof(BITMAPFILEHEADER), 1, outptr);
    fwrite(&bi, sizeof(BITMAPINFOHEADER), 1, outptr);

    for (int i = 0; i < height; i++)
    {
        fwrite(image[i], sizeof(RGBTRIPLE), width, outptr);
        for (int k = 0; k < padding; k++)
            fputc(0x00, outptr);
    }

    free(image);
    fclose(outptr);

    return 0;
}