#include "../bmp.h"


void apply_functional_sequentially(int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

void apply_functional_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

void apply_functional_sequentiallyMPI(int height, int width, int from, int to, RGBTRIPLE image[height][width],
                                      void (*filter_function)(int *, int *, int *));

int apply_functional(int x, int y, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

void grayscale(int *r, int *g, int *b);

void inversion(int *r, int *g, int *b);


