#ifndef _PNG_RW_H_
#define _PNG_RW_H_
#include <png.h>
#include <zlib.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
int read_png(const char*, unsigned char**, unsigned*, unsigned*, unsigned*);
void write_png(const char*, png_bytep, const unsigned, const unsigned, const unsigned);
#endif