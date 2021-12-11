#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

clock_t get_clock_time();

double benchmark_time(clock_t start, clock_t end);

double get_time();

double get_elapsed_time(struct timeval begin, struct  timeval end);