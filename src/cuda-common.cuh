/*
 * Het-SCD: High Quality Community Detection on Heterogenous Platforms 
 * Copyright (C) 2015, S. Heldens
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CUDA_COMMON_HPP
#define CUDA_COMMON_HPP

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <functional>
#include <iostream>

#include "common.hpp"

#define CUDA_CHECK(msg, res) \
    __cuda_check(__FILE__, __func__, __LINE__, (msg), (res))

#define CUDA_CALL(func, ...) \
    CUDA_CHECK(#func, func(__VA_ARGS__));

#define CUDA_CHECK_LAST(msg) \
   CUDA_CHECK((msg), cudaPeekAtLastError())

#define CUDA_LAUNCH(ctx, global_size, kernel, ...) \
    do { \
        dim3 __grid(ceil_div(global_size, block_size), 1); \
        dim3 __block(block_size, 1); \
        if (__grid.x >= 65536) __grid.x = __grid.y = ceil(sqrt(__grid.x)); \
        kernel<<<__grid, __block, 0, (ctx)->Stream()>>>(MAP(strip_mgpu_mem, __VA_ARGS__)); \
        CUDA_CHECK_LAST(#kernel); \
    } while (0)

#define CUDA_SYNC() \
    CUDA_CALL(cudaDeviceSynchronize)

#define CUDA_FILL(ctx, ptr, count, value) \
    CUDA_LAUNCH(ctx, count, kernel_fill, count, ptr, value)

#define CUDA_CLEAR(ctx, ptr, count) \
    CUDA_CALL(cudaMemsetAsync, strip_mgpu_mem(ptr), 0, (count) * sizeof(*strip_mgpu_mem(ptr)), (ctx)->Stream())


template <typename T>
inline static T* strip_mgpu_mem(MGPU_MEM(T) t) {
    return t->get();
}

template <typename T>
inline static T strip_mgpu_mem(T t) {
    return t;
}

inline static cudaError_t __cuda_check(const char *file, const char *func,
        int line, const char *msg, cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA fatal error: " << file << ":" << func << ":"
            << line << ": " << msg << ": "
            << cudaGetErrorString(code) << std::endl;
        exit(code);
    }

    return code;
}

__device__ int get_global_id(void) {
      return blockIdx.x * blockDim.x + threadIdx.x +
             (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x;
}


__device__ int get_global_size(void) {
      return blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}

template <typename T>
__global__ void kernel_fill(const int n, T *ptr, T val) {
    int i = get_global_id();
    if (i < n) ptr[i] = val;
}

__device__ __host__ bool operator==(int2 a, int2 b) {
    return a.x == b.x && a.y == b.y;
}

struct less_second : public std::binary_function<int2, int2, int2> {
    __device__ __host__ inline bool operator() (int2 a, int2 b) {
        return a.y < b.y;
    }
};

struct max_second_min_first : public std::binary_function<int2, int2, int2> {
    __device__ __host__ inline int2 operator() (int2 a, int2 b) {
        float fa = *(float*) &(a.y);
        float fb = *(float*) &(b.y);

        return fa > fb || (fa == fb && a.x < b.x) ? a : b;
    }
};

class int2_zip_iterator : public std::iterator_traits<const int2*> {
public:
    __host__ __device__ int2_zip_iterator(int *it1, int *it2) :
        _it1(it1), _it2(it2) { }

    __host__ __device__ int2 operator[](ptrdiff_t i) {
        return make_int2(*(_it1 + i), *(_it2 + i));
    }

    __host__ __device__ int2 operator*() {
        return make_int2(*_it1, *_it2);
    }

    __host__ __device__ int2_zip_iterator operator+(ptrdiff_t diff) {
        return int2_zip_iterator(_it1 + diff, _it2 + diff);
    }

    __host__ __device__ int2_zip_iterator operator-(ptrdiff_t diff) {
        return int2_zip_iterator(_it1 - diff, _it2 - diff);
    }

    __host__ __device__ int2_zip_iterator& operator+=(ptrdiff_t diff) {
        _it1 += diff;
        _it2 += diff;
        return *this;
    }

    __host__ __device__ int2_zip_iterator& operator-=(ptrdiff_t diff) {
        _it1 -= diff;
        _it2 -= diff;
        return *this;
    }

private:
    int *_it1;
    int *_it2;
};

static int2_zip_iterator make_int2_zip_iterator(int *it1, int *it2) {
    return int2_zip_iterator(it1, it2);
}

#endif
