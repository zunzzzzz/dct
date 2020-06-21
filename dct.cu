#include "png_rw.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ctime>

__global__
void DCT(unsigned char* src_img, double* dct_img, unsigned height, unsigned width, unsigned channels)
{
    // set up cosine table
    double cosine[8][8];
    const double inv16 = 1.0 / 16.0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cosine[j][i] = cos(M_PI * j * (2.0 * i + 1) * inv16);
        }
    }
    for(int width_iter = threadIdx.x * 8; width_iter < width; width_iter += blockDim.x * 8) {
        int height_iter = blockIdx.x * 8;
        for(int u = width_iter; u < width_iter + 8; u++) {
            for(int v = height_iter; v < height_iter + 8; v++) {

                double cu, cv;
                if(u % 8 == 0) cu = 1 / sqrtf(2);
                else cu = 1;
                if(v % 8 == 0) cv = 1 / sqrtf(2);
                else cv = 1;
                for(int x = width_iter; x < width_iter + 8; x++) {
                    for(int y = height_iter; y <  height_iter + 8; y++) {

                        double R, G, B;
                        R = cu * cv * src_img[channels * (width * y + x) + 0] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        G = cu * cv * src_img[channels * (width * y + x) + 1] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        B = cu * cv * src_img[channels * (width * y + x) + 2] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        dct_img[channels * (width * v + u) + 0] += R;
                        dct_img[channels * (width * v + u) + 1] += G;
                        dct_img[channels * (width * v + u) + 2] += B;
                    }
                }
            }
        }
    }
}
__global__
void IDCT(double* dct_img, unsigned char* dst_img, unsigned height, unsigned width, unsigned channels)
{
    // set up cosine table
    double cosine[8][8];
    const double inv16 = 1.0 / 16.0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cosine[j][i] = cos(M_PI * j * (2.0 * i + 1) * inv16);
        }
    }
    for(int width_iter = threadIdx.x * 8; width_iter < width; width_iter += blockDim.x * 8) {
        int height_iter = blockIdx.x * 8;
        for(int x = width_iter; x < width_iter + 8; x++) {
            for(int y = height_iter; y < height_iter + 8; y++) {

                double cu, cv;
                for(int u = width_iter; u < width_iter + 8; u++) {
                    for(int v = height_iter; v < height_iter + 8; v++) {

                        if(u % 8 == 0) cu = 1 / sqrtf(2);
                        else cu = 1;
                        if(v % 8 == 0) cv = 1 / sqrtf(2);
                        else cv = 1;
                        double R, G, B;
                        R = cu * cv * dct_img[channels * (width * v + u) + 0] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        G = cu * cv * dct_img[channels * (width * v + u) + 1] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        B = cu * cv * dct_img[channels * (width * v + u) + 2] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        dst_img[channels * (width * y + x) + 0] += R;
                        dst_img[channels * (width * y + x) + 1] += G;
                        dst_img[channels * (width * y + x) + 2] += B;
                    }
                }
            }
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 3);
    clock_t start, end;
    unsigned height, width, channels;
    unsigned char* src_img_HOST = NULL;
    unsigned char* dst_img_HOST = NULL;
    unsigned char* src_img_GPU = NULL;
    unsigned char* dst_img_GPU = NULL;
    double* dct_img_HOST = NULL;
    double* dct_img_GPU = NULL;
    size_t threads_per_block, num_of_blocks;

    start = clock();
    read_png(argv[1], &src_img_HOST, &height, &width, &channels);
    assert(channels == 3);
    // allocate memory
    dst_img_HOST = new unsigned char[width * height * channels];
    memset(dst_img_HOST, 0, sizeof(dst_img_HOST));
    dct_img_HOST = new double[width * height * channels]();

    cudaMalloc(&dst_img_GPU, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&src_img_GPU, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&dct_img_GPU, height * width * channels * sizeof(double));
    // memory copy 
    cudaMemcpy(src_img_GPU, src_img_HOST, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dct_img_GPU, dct_img_HOST, height * width * channels * sizeof(double), cudaMemcpyHostToDevice);
    
    printf("width = %d, height = %d, channel = %d\n", width, height, channels);
    
    threads_per_block = 256;
    num_of_blocks = height / 8;

    DCT<<<num_of_blocks, threads_per_block>>>(src_img_GPU, dct_img_GPU, height, width, channels);
    cudaDeviceSynchronize();

    IDCT<<<num_of_blocks, threads_per_block>>>(dct_img_GPU, dst_img_GPU, height, width, channels);
    cudaDeviceSynchronize();
    // copy back
    cudaMemcpy(dst_img_HOST, dst_img_GPU, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    write_png(argv[2], dst_img_HOST, height, width, channels);
    end = clock();
    printf("%f\n", ((double)(end - start) / CLOCKS_PER_SEC));
    return 0;
}
