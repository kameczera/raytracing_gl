#ifndef COLOR_H
#define COLOR_H

#include "vec3.cuh"

using color = vec3;

__device__ float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ uchar4 write_color(const color& pixel_color) {
    return make_uchar4(
        static_cast<unsigned char>(255.999f * clamp(pixel_color.x(), 0.0f, 1.0f)),
        static_cast<unsigned char>(255.999f * clamp(pixel_color.y(), 0.0f, 1.0f)),
        static_cast<unsigned char>(255.999f * clamp(pixel_color.z(), 0.0f, 1.0f)),
        255
    );
}

#endif