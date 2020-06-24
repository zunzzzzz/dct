#include "png_rw.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ctime>


int main(int argc, char** argv) {
    assert(argc == 3);
    clock_t start, end;
    unsigned height, width, channels;
    unsigned char* src_img_HOST = NULL;
    double* dct_img_HOST = NULL;
    unsigned char* dst_img_HOST = NULL;
    size_t threads_per_block = 256;
    size_t num_of_blocks = 32 * 20;

    start = clock();
    read_png(argv[1], &src_img_HOST, &height, &width, &channels);
    assert(channels == 3);
    // allocate dst memory
    dst_img_HOST = new unsigned char[width * height * channels];
    memset(dst_img_HOST, 0, sizeof(dst_img_HOST));
    dct_img_HOST = new double[width * height * channels]();

    // set up cosine table
    double cosine[8][8];
    const double inv16 = 1.0 / 16.0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cosine[j][i] = cos(M_PI * j * (2.0 * i + 1) * inv16);
        }
    }
    printf("width = %d, height = %d, channel = %d\n", width, height, channels);
    // rgb to ycbcr
    for(int width_iter = 0; width_iter < width; width_iter++) {
        for(int height_iter = 0; height_iter < height; height_iter++) {
            double R, G, B, Y, Cb, Cr;
            R = (double) src_img_HOST[channels * (width * height_iter + width_iter) + 0];
            G = (double) src_img_HOST[channels * (width * height_iter + width_iter) + 1];
            B = (double) src_img_HOST[channels * (width * height_iter + width_iter) + 2];
            // printf("before = %d\n", src_img_HOST[channels * (width * height_iter + width_iter) + 0]);
            // printf("tmp = %f\n", 0.299 * R + 0.578 * G + 0.114 * B);
            Y = 0.257 * R + 0.564 * G + 0.098 * B + 16;
            Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128;
            Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128;
            src_img_HOST[channels * (width * height_iter + width_iter) + 0] = Y;
            src_img_HOST[channels * (width * height_iter + width_iter) + 1] = Cb;
            src_img_HOST[channels * (width * height_iter + width_iter) + 2] = Cr;
        }
    }
    // dct
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int u = width_iter; u < width_iter + 8; u++) {
                for(int v = height_iter; v < height_iter + 8; v++) {

                    double cu, cv;
                    if(u % 8 == 0) cu = 1 / sqrt(2);
                    else cu = 1;
                    if(v % 8 == 0) cv = 1 / sqrt(2);
                    else cv = 1;
                    for(int x = width_iter; x < width_iter + 8; x++) {
                        for(int y = height_iter; y <  height_iter + 8; y++) {

                            double R, G, B;
                            R = cu * cv * src_img_HOST[channels * (width * y + x) + 0] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            G = cu * cv * src_img_HOST[channels * (width * y + x) + 1] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            B = cu * cv * src_img_HOST[channels * (width * y + x) + 2] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            dct_img_HOST[channels * (width * v + u) + 0] += R;
                            dct_img_HOST[channels * (width * v + u) + 1] += G;
                            dct_img_HOST[channels * (width * v + u) + 2] += B;
                        }
                    }
                }
            }
        }
    }
    // quantization part
    unsigned luminance[8][8] =
    {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 95 ,112, 100, 103, 99
    };
     unsigned chrominance[8][8] =
    {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 993,99, 99, 99,
        99, 99, 99, 99 ,992,99, 99, 99
    };
    // quantize
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int u = width_iter; u < width_iter + 8; u++) {
                for(int v = height_iter; v < height_iter + 8; v++) {
                    dct_img_HOST[channels * (width * v + u) + 0] = abs(round(dct_img_HOST[channels * (width * v + u) + 0] / luminance[u % 8][v % 8]));
                    dct_img_HOST[channels * (width * v + u) + 1] = abs(round(dct_img_HOST[channels * (width * v + u) + 1] / chrominance[u % 8][v % 8]));
                    dct_img_HOST[channels * (width * v + u) + 2] = abs(round(dct_img_HOST[channels * (width * v + u) + 2] / chrominance[u % 8][v % 8]));
                    // if(width_iter == 0 && height_iter == 0) printf("%f ", dct_img_HOST[channels * (width * v + u) + 0]);
                }
                // if(width_iter == 0 && height_iter == 0) printf("\n");
            }
        }
    }
    // dequantize
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int u = width_iter; u < width_iter + 8; u++) {
                for(int v = height_iter; v < height_iter + 8; v++) {
                    dct_img_HOST[channels * (width * v + u) + 0] = abs(round(dct_img_HOST[channels * (width * v + u) + 0] * luminance[u % 8][v % 8]));
                    dct_img_HOST[channels * (width * v + u) + 1] = abs(round(dct_img_HOST[channels * (width * v + u) + 1] * chrominance[u % 8][v % 8]));
                    dct_img_HOST[channels * (width * v + u) + 2] = abs(round(dct_img_HOST[channels * (width * v + u) + 2] * chrominance[u % 8][v % 8]));
                    // if(width_iter == 0 && height_iter == 0) printf("%f ", dct_img_HOST[channels * (width * v + u) + 0]);
                }
                // if(width_iter == 0 && height_iter == 0) printf("\n");
            }
        }
    }
    // idct
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int x = width_iter; x < width_iter + 8; x++) {
                for(int y = height_iter; y < height_iter + 8; y++) {

                    double cu, cv;
                    for(int u = width_iter; u < width_iter + 8; u++) {
                        for(int v = height_iter; v < height_iter + 8; v++) {

                            if(u % 8 == 0) cu = 1 / sqrt(2);
                            else cu = 1;
                            if(v % 8 == 0) cv = 1 / sqrt(2);
                            else cv = 1;
                            double R, G, B;
                            R = cu * cv * dct_img_HOST[channels * (width * v + u) + 0] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            G = cu * cv * dct_img_HOST[channels * (width * v + u) + 1] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            B = cu * cv * dct_img_HOST[channels * (width * v + u) + 2] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                            dst_img_HOST[channels * (width * y + x) + 0] += R;
                            dst_img_HOST[channels * (width * y + x) + 1] += G;
                            dst_img_HOST[channels * (width * y + x) + 2] += B;
                        }
                    }
                }
            }
        }    
    }
    // ycbcr to rgb
    for(int width_iter = 0; width_iter < width; width_iter++) {
        for(int height_iter = 0; height_iter < height; height_iter++) {
            double Y, Cb, Cr, R, G, B;
            Y = src_img_HOST[channels * (width * height_iter + width_iter) + 0];
            Cb = src_img_HOST[channels * (width * height_iter + width_iter) + 1];
            Cr = src_img_HOST[channels * (width * height_iter + width_iter) + 2];
            R = 1.164 * (Y - 16) + 1.596 * (Cr - 128);
            G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128);
            B = 1.164 * (Y - 16) + 2.017 * (Cb - 128);
            src_img_HOST[channels * (width * height_iter + width_iter) + 0] = R;
            src_img_HOST[channels * (width * height_iter + width_iter) + 1] = G;
            src_img_HOST[channels * (width * height_iter + width_iter) + 2] = B;
        }
    }

    
    write_png(argv[2], src_img_HOST, height, width, channels);
    end = clock();
    printf("%f\n", ((double)(end - start) / CLOCKS_PER_SEC));
    return 0;
}
