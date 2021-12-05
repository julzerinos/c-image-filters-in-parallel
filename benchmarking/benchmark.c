#include "benchmark.h"

clock_t get_clock_time()
{
    return clock();
}

double benchmark_time(clock_t start, clock_t end)
{
    return (double) (end - start) / CLOCKS_PER_SEC;
}