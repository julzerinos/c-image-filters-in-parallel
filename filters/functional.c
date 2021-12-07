#include <math.h>
#include <threads.h>
#include <assert.h>


#include "functional.h"

// Implement parallel functions here

void apply_functional_sequentially(int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *))
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            apply_functional(i, j, height, width, image, filter_function);
}

int apply_functional(int x, int y, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *))
{
    int r = image[x][y].rgbtRed, g = image[x][y].rgbtGreen, b = image[x][y].rgbtBlue;
    
    filter_function(&r, &g, &b);

    image[x][y].rgbtRed = r;
    image[x][y].rgbtGreen = g;
    image[x][y].rgbtBlue = b;
    return 0;
}

void grayscale(int *r, int *g, int *b)
{
    BYTE grayValue = .3 * (*r) + .59 * (*g) + .11 * (*b);
    *r = grayValue;
    *g = grayValue;
    *b = grayValue;
}

struct threadParams
{
    int height;
    int width;
    RGBTRIPLE* image;
    void (*filter_function)(int *, int *, int *);
}ThreadParams;

void apply_functional_parallelly(int thread_count, int height, int width, RGBTRIPLE image[height][width], void (*filter_function)(int *, int *, int *))
{
    assert(thread_count <= height);
    int rows_for_thread = height / thread_count;
    assert(rows_for_thread * thread_count == height); 

    thrd_t threads[thread_count];


    struct threadParams params;
    for(int i=0, current_row=0; i<thread_count; ++i, current_row+=rows_for_thread)
    {
        params.height = rows_for_thread;
        params.width = width;
        params.image = *image;
        params.filter_function = filter_function;
        thrd_create(&threads[i], ((int(*)())apply_functional_sequentially), &params); 
    }

    for(int i=0; i<thread_count; ++i)
    {
        thrd_join(threads[i], NULL); 
    }
}

