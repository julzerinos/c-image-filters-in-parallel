
#include "../bmp.h"


void apply_functional_sequentially(int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

void apply_functional_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

int apply_functional(int x, int y, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *));

void grayscale(int *r, int *g, int *b);

void castingFunction(void* arg);