#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "convolution.h"
#include <mpi.h>
#include <openacc.h>
void create_blur_kernel(int dimension, double kernel[dimension][dimension])
// https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
{

    MPI_Recv( NULL, 100, MPI_DOUBLE, 1, 19, MPI_COMM_WORLD, NULL );

    double sigma = 1;
    int W = dimension;
    double mean = W / 2;
    double sum = 0.0;
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y) {
            kernel[x][y] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) /
                           (2 * M_PI * sigma * sigma);

            // Accumulate the kernel values
            sum += kernel[x][y];
        }

    // Normalize the kernel
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y)
            kernel[x][y] /= sum;
}

void create_edge_detection_kernel(int dimension, double kernel[dimension][dimension]) {
    kernel[0][0] = 0;
    kernel[0][1] = 1;
    kernel[0][2] = 0;
    kernel[1][0] = 1;
    kernel[1][1] = -4;
    kernel[1][2] = 1;
    kernel[2][0] = 0;
    kernel[2][1] = 1;
    kernel[2][2] = 0;
}


void apply_convolution_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension,
                                    double kernel[kernel_dimension][kernel_dimension]) {
    RGBTRIPLE temp_img[height][width];

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            apply_convolution(i, j, height, width, image, temp_img, kernel_dimension, kernel);

    for (int i = 0; i < height; i++) // copying contents of new image to the original image
    {
        for (int j = 0; j < width; j++) {
            image[i][j].rgbtBlue = temp_img[i][j].rgbtBlue;
            image[i][j].rgbtGreen = temp_img[i][j].rgbtGreen;
            image[i][j].rgbtRed = temp_img[i][j].rgbtRed;
        }
    }
}

void apply_convolution(int x, int y, int height, int width, RGBTRIPLE image[height][width],
                       RGBTRIPLE target_image[height][width], int kernel_dimension,
                       double kernel[kernel_dimension][kernel_dimension]) {
    float totalRed = 0.0;
    float totalGreen = 0.0;
    float totalBlue = 0.0;

    int kernel_spread = kernel_dimension / 2;
    for (int i = -kernel_spread; i <= kernel_spread; i++)
        for (int j = -kernel_spread; j <= kernel_spread; j++) {
            int xIndex = x + i;
            int yIndex = y + j;

            if (xIndex >= width || xIndex < 0 || yIndex >= height || yIndex < 0)
                continue; // Assume 0 value for out of bound pixels

            totalRed += kernel[i + kernel_spread][j + kernel_spread] * image[xIndex][yIndex].rgbtRed;
            totalGreen += kernel[i + kernel_spread][j + kernel_spread] * image[xIndex][yIndex].rgbtGreen;
            totalBlue += kernel[i + kernel_spread][j + kernel_spread] * image[xIndex][yIndex].rgbtBlue;
        }

    target_image[x][y].rgbtBlue = round(totalBlue);
    target_image[x][y].rgbtRed = round(totalRed);
    target_image[x][y].rgbtGreen = round(totalGreen);
}

void apply_convolution_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width],
                                  int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]) {
    RGBTRIPLE temp_img[height][width];

    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
#pragma omp parallel shared(image, height, width, temp_img, kernel_dimension, kernel) default(none)
    {
#pragma omp for
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                apply_convolution(i, j, height, width, image, temp_img, kernel_dimension, kernel);

    }
    for (int i = 0; i < height; i++) // copying contents of new image to the original image
    {
        for (int j = 0; j < width; j++) {
            image[i][j].rgbtBlue = temp_img[i][j].rgbtBlue;
            image[i][j].rgbtGreen = temp_img[i][j].rgbtGreen;
            image[i][j].rgbtRed = temp_img[i][j].rgbtRed;
        }
    }
}

void apply_convolution_parallellyMPI(int thread_count, int height, int width, RGBTRIPLE image[height][width],
                                  int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]) {
    RGBTRIPLE temp_img[height][width];

    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
#pragma omp parallel shared(image, height, width, temp_img, kernel_dimension, kernel) default(none)
    {
#pragma omp for
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                apply_convolution(i, j, height, width, image, temp_img, kernel_dimension, kernel);

    }
    for (int i = 0; i < height; i++) // copying contents of new image to the original image
    {
        for (int j = 0; j < width; j++) {
            image[i][j].rgbtBlue = temp_img[i][j].rgbtBlue;
            image[i][j].rgbtGreen = temp_img[i][j].rgbtGreen;
            image[i][j].rgbtRed = temp_img[i][j].rgbtRed;
        }
    }
}


void
edge_detection_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], int kernel_dimension,
                          double kernel[kernel_dimension][kernel_dimension]) {
    create_blur_kernel(kernel_dimension, kernel);
    apply_convolution_parallelly(thread_count, height, width, image, kernel_dimension, kernel);
    create_edge_detection_kernel(kernel_dimension, kernel);
    apply_convolution_parallelly(thread_count, height, width, image, kernel_dimension, kernel);
}


void edge_detection_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension,
                                 double kernel[kernel_dimension][kernel_dimension]) {
    create_blur_kernel(kernel_dimension, kernel);
    apply_convolution_sequentially(height, width, image, kernel_dimension, kernel);
    create_edge_detection_kernel(kernel_dimension, kernel);
    apply_convolution_sequentially(height, width, image, kernel_dimension, kernel);
}

void blur_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension,
                       double kernel[kernel_dimension][kernel_dimension]) {
    create_blur_kernel(kernel_dimension, kernel);
    apply_convolution_sequentially(height, width, image, kernel_dimension, kernel);
}

void blur_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], int kernel_dimension,
                     double kernel[kernel_dimension][kernel_dimension]) {
    create_blur_kernel(kernel_dimension, kernel);
    apply_convolution_parallelly(thread_count, height, width, image, kernel_dimension, kernel);
}