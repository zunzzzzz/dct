#include "png_rw.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>

class RLE_DATA
{
public:
    unsigned consecutive_zero = 0;
    int value;
};
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
        unsigned tmp_channel_one[8][8],  tmp_channel_two[8][8], tmp_channel_three[8][8];
        for(int x = width_iter; x < width_iter + 8; x++) {
            for(int y = height_iter; y <  height_iter + 8; y++) {
                tmp_channel_one[x % 8][y % 8] = src_img[channels * (width * y + x) + 0];
                tmp_channel_two[x % 8][y % 8] = src_img[channels * (width * y + x) + 1];
                tmp_channel_three[x % 8][y % 8] = src_img[channels * (width * y + x) + 2];
            }
        }
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
                        R = cu * cv * tmp_channel_one[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        G = cu * cv * tmp_channel_two[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        B = cu * cv * tmp_channel_three[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
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
        double tmp_channel_one[8][8],  tmp_channel_two[8][8], tmp_channel_three[8][8];
        for(int x = width_iter; x < width_iter + 8; x++) {
            for(int y = height_iter; y <  height_iter + 8; y++) {
                tmp_channel_one[x % 8][y % 8] = dct_img[channels * (width * y + x) + 0];
                tmp_channel_two[x % 8][y % 8] = dct_img[channels * (width * y + x) + 1];
                tmp_channel_three[x % 8][y % 8] = dct_img[channels * (width * y + x) + 2];
            }
        }
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
                        R = cu * cv * tmp_channel_one[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        G = cu * cv * tmp_channel_two[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
                        B = cu * cv * tmp_channel_three[x % 8][y % 8] * cosine[u % 8][x % 8] * cosine[v % 8][y % 8] / 4;
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
    threads_per_block = 256;
    num_of_blocks = height / 8;
    cudaMemcpy(src_img_GPU, src_img_HOST, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dct_img_GPU, dct_img_HOST, height * width * channels * sizeof(double), cudaMemcpyHostToDevice);
    DCT<<<num_of_blocks, threads_per_block>>>(src_img_GPU, dct_img_GPU, height, width, channels);
    cudaMemcpy(dct_img_HOST, dct_img_GPU, height * width * channels * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
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
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99 ,99, 99, 99, 99
    };
    // quantize
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int u = width_iter; u < width_iter + 8; u++) {
                for(int v = height_iter; v < height_iter + 8; v++) {
                    dct_img_HOST[channels * (width * v + u) + 0] = round(dct_img_HOST[channels * (width * v + u) + 0] / luminance[u % 8][v % 8]);
                    dct_img_HOST[channels * (width * v + u) + 1] = round(dct_img_HOST[channels * (width * v + u) + 1] / chrominance[u % 8][v % 8]);
                    dct_img_HOST[channels * (width * v + u) + 2] = round(dct_img_HOST[channels * (width * v + u) + 2] / chrominance[u % 8][v % 8]);
                    // if(width_iter == 0 && height_iter == 0) printf("%f ", dct_img_HOST[channels * (width * v + u) + 0]);
                }
                // if(width_iter == 0 && height_iter == 0) printf("\n");
            }
        }
    }
    // DPCM
    unsigned y_first, cb_first, cr_first;
    unsigned y_pre, cb_pre, cr_pre;
    unsigned y_dpcm[width * height / 64], cb_dpcm[width * height / 64], cr_dpcm[width * height / 64];
    int count = 0;
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8, count++) {
            if(width_iter == 0 && height_iter == 0) {
                y_first = dct_img_HOST[channels * (width * height_iter + width_iter) + 0];
                cb_first = dct_img_HOST[channels * (width * height_iter + width_iter) + 1];
                cr_first = dct_img_HOST[channels * (width * height_iter + width_iter) + 2];
                y_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 0];
                cb_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 1];
                cr_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 2];
            }
            else{
                y_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 0] - y_pre;
                cb_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 1] - cb_pre;
                cr_dpcm[count] = dct_img_HOST[channels * (width * height_iter + width_iter) + 2] - cr_pre;
            }
            y_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 0];
            cb_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 1];
            cr_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 2];
        }
    }
    // RLE
    int total_bits = width * height * channels * 8;
    int accu_bits = 0;
    double compression_ratio;
    int zig_zag[64] =
    {
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 
        5, 12, 19, 26, 33, 40, 48, 
        41, 34, 27, 20, 13, 6, 7,
        14, 24, 28, 35, 42, 49, 56, 
        57, 50, 43, 36, 29, 22, 15,
        23, 30, 37, 44, 51, 58, 59,
        52, 45, 38, 31, 39, 46, 53,
        60, 61, 54, 47, 55, 62, 63 
    };
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {
            std::vector<RLE_DATA> y_RLE;
            std::vector<RLE_DATA> cb_RLE;
            std::vector<RLE_DATA> cr_RLE;
            RLE_DATA y_tmp, cb_tmp, cr_tmp;
            // encoding
            for(int iter = 0; iter < 64; iter++) {
                int u = zig_zag[iter] % 8;
                int v = zig_zag[iter] / 8;
                // y
                if(u % 8 == 0 && v % 8 == 0) {
                    // do nothing
                }
                else if(dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 0] == 0) {
                    y_tmp.consecutive_zero++;
                }
                else {
                    y_tmp.value = dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 0];
                    y_RLE.push_back(y_tmp);
                    y_tmp.consecutive_zero = 0;
                }
                // cb
                if(u % 8 == 0 && v % 8 == 0) {
                    // do nothing
                }
                else if(dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 1] == 0) {
                    cb_tmp.consecutive_zero++;
                }
                else {
                    cb_tmp.value = dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 1];
                    cb_RLE.push_back(cb_tmp);
                    cb_tmp.consecutive_zero = 0;
                }
                // cr
                if(u % 8 == 0 && v % 8 == 0) {
                    // do nothing
                }
                else if(dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 2] == 0) {
                    cr_tmp.consecutive_zero++;
                }
                else {
                    cr_tmp.value = dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 2];
                    cr_RLE.push_back(cr_tmp);
                    cr_tmp.consecutive_zero = 0;
                }
            }
            int partial_bits = 40 * (y_RLE.size() + cb_RLE.size() + cr_RLE.size());
            accu_bits += partial_bits;
            // decoding
            // reset to zero to test 
            for(int iter = 0; iter < 64; iter++) {
                int u = zig_zag[iter] % 8;
                int v = zig_zag[iter] / 8;
                
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 0] = 0;
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 1] = 0;
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 2] = 0;
            }
            // y
            int index = 0;
            for(int iter = 0; iter < y_RLE.size(); iter++) {
                index += y_RLE[iter].consecutive_zero + 1;
                int u = zig_zag[index] % 8;
                int v = zig_zag[index] / 8;
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 0] = y_RLE[iter].value;
            }
            // cb
            index = 0;
            for(int iter = 0; iter < cb_RLE.size(); iter++) {
                index += cb_RLE[iter].consecutive_zero + 1;
                int u = zig_zag[index] % 8;
                int v = zig_zag[index] / 8;
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 1] = cb_RLE[iter].value;
            }
            // cr
            index = 0;
            for(int iter = 0; iter < cr_RLE.size(); iter++) {
                index += cr_RLE[iter].consecutive_zero + 1;
                int u = zig_zag[index] % 8;
                int v = zig_zag[index] / 8;
                dct_img_HOST[channels * (width * (v + height_iter) + (u + width_iter)) + 2] = cr_RLE[iter].value;
            }
        }
    }
    compression_ratio = (double) total_bits / (double) (accu_bits + 24);
    printf("compression ratio = %f\n", compression_ratio);
    // recover DPCM
    count = 0;
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8, count++) {
            if(width_iter == 0 && height_iter == 0) {
                dct_img_HOST[channels * (width * height_iter + width_iter) + 0] = y_first;
                dct_img_HOST[channels * (width * height_iter + width_iter) + 1] = cb_first;
                dct_img_HOST[channels * (width * height_iter + width_iter) + 2] = cr_first;
            }
            else{
                dct_img_HOST[channels * (width * height_iter + width_iter) + 0] = y_pre + y_dpcm[count];
                dct_img_HOST[channels * (width * height_iter + width_iter) + 1] = cb_pre + cb_dpcm[count];
                dct_img_HOST[channels * (width * height_iter + width_iter) + 2] = cr_pre + cr_dpcm[count];
            }
            y_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 0];
            cb_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 1];
            cr_pre = dct_img_HOST[channels * (width * height_iter + width_iter) + 2];
        }
    }
    // dequantize
    for(int width_iter = 0; width_iter < width; width_iter += 8) {
        for(int height_iter = 0; height_iter < height; height_iter += 8) {

            for(int u = width_iter; u < width_iter + 8; u++) {
                for(int v = height_iter; v < height_iter + 8; v++) {
                    dct_img_HOST[channels * (width * v + u) + 0] = round(dct_img_HOST[channels * (width * v + u) + 0] * luminance[u % 8][v % 8]);
                    dct_img_HOST[channels * (width * v + u) + 1] = round(dct_img_HOST[channels * (width * v + u) + 1] * chrominance[u % 8][v % 8]);
                    dct_img_HOST[channels * (width * v + u) + 2] = round(dct_img_HOST[channels * (width * v + u) + 2] * chrominance[u % 8][v % 8]);
                    // if(width_iter == 0 && height_iter == 0) printf("%f ", dct_img_HOST[channels * (width * v + u) + 0]);
                }
                // if(width_iter == 0 && height_iter == 0) printf("\n");
            }
        }
    }
    cudaMemcpy(dct_img_GPU, dct_img_HOST, height * width * channels * sizeof(double), cudaMemcpyHostToDevice);
    IDCT<<<num_of_blocks, threads_per_block>>>(dct_img_GPU, dst_img_GPU, height, width, channels);
    cudaMemcpy(dst_img_HOST, dst_img_GPU, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // ycbcr to rgb
    for(int width_iter = 0; width_iter < width; width_iter++) {
        for(int height_iter = 0; height_iter < height; height_iter++) {
            double Y, Cb, Cr, R, G, B;
            Y = dst_img_HOST[channels * (width * height_iter + width_iter) + 0];
            Cb = dst_img_HOST[channels * (width * height_iter + width_iter) + 1];
            Cr = dst_img_HOST[channels * (width * height_iter + width_iter) + 2];
            R = 1.164 * (Y - 16) + 1.596 * (Cr - 128);
            G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128);
            B = 1.164 * (Y - 16) + 2.017 * (Cb - 128);
            dst_img_HOST[channels * (width * height_iter + width_iter) + 0] = R;
            dst_img_HOST[channels * (width * height_iter + width_iter) + 1] = G;
            dst_img_HOST[channels * (width * height_iter + width_iter) + 2] = B;
        }
    }
    write_png(argv[2], dst_img_HOST, height, width, channels);
    end = clock();
    printf("execution time = %f\n", ((double)(end - start) / CLOCKS_PER_SEC));
    return 0;
}
