cmake_minimum_required(VERSION 3.13)
project(c-image-filters-in-parallel)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -fopenmp")
set(CMAKE_C_STANDARD 11)

include_directories(./benchmarking)
include_directories(./filters)
include_directories(.)

add_executable(c-image-filters-in-parallel main.c
        benchmarking/benchmark.c
        filters/convolution.c
        filters/functional.c)