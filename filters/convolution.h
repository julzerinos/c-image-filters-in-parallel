#include "../bmp.h"

void create_blur_kernel(int dimension, double kernel[dimension][dimension]);

void create_edge_detection_kernel(int dimension, double kernel[dimension][dimension]);

void apply_convolution_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void apply_convolution(int x, int y, int height, int width, RGBTRIPLE image[height][width], RGBTRIPLE target_image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void apply_convolution_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void blur_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void blur_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void edge_detection_sequentially(int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);

void edge_detection_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], int kernel_dimension, double kernel[kernel_dimension][kernel_dimension]);