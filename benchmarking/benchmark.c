#include "benchmark.h"

clock_t get_clock_time()
{
    return clock();
}

double benchmark_time(clock_t start, clock_t end)
{
    return (double) (end - start) / CLOCKS_PER_SEC;
}


double get_elapsed_time(struct timeval begin, struct  timeval end)
{
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    return elapsed;
}

