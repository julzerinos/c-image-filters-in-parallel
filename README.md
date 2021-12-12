## Deployment

`gcc main.c filters/functional.c filters/convolution.c benchmarking/benchmark.c -o main -lm -fopenmp`

## Usage

`./main -[c|f] -[0|1] input_bmp output_bmp`

 * f - functional filters 
   * 0 - grayscale
   * 1 - inversion
 * c - convolutional/kernel filters
   * 0 - blur
   * 1 - edge detection