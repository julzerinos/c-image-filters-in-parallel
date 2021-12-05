#include <math.h>

#include "functional.h"

// Implement parallel functions here

void apply_functional_sequentially(int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *))
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            apply_functional(i, j, height, width, image, filter_function);
}

void apply_functional(int x, int y, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *))
{
    int r = image[x][y].rgbtRed, g = image[x][y].rgbtGreen, b = image[x][y].rgbtBlue;
    
    filter_function(&r, &g, &b);

    image[x][y].rgbtRed = r;
    image[x][y].rgbtGreen = g;
    image[x][y].rgbtBlue = b;
}

void grayscale(int *r, int *g, int *b)
{
    BYTE grayValue = .3 * (*r) + .59 * (*g) + .11 * (*b);
    *r = grayValue;
    *g = grayValue;
    *b = grayValue;
}
