#ifndef VEC3_CUH
#define VEC3_CUH

#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

class vec3 {
  public:
    double e[3];

    __host__ __device__ vec3() : e{0,0,0} {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    __host__ __device__ double length() const {
        return std::sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ bool near_zero() {
        int s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }
};

using point3 = vec3;

__device__ inline vec3 random(curandState* local_state) {
    return vec3(curand_uniform(local_state), curand_uniform(local_state), curand_uniform(local_state));
}

__device__ inline double random_double(curandState* local_state) {
    return curand_uniform(local_state);
}

__host__ inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif