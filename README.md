## Deployment

`gcc main.c filters/functional.c filters/convolution.c benchmarking/benchmark.c -o main -lm`

## Usage

`./main -[g|b] input_bmp output_bmp`

 * g - grayscale (functional)
 * b - blur (convolution/kernel)
