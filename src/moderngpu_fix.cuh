#pragma once

#include <moderngpu.cuh>
#include <iterator>

namespace mgpu {

/**
 * The default implementation of mgpu::ldg accepts only "const T*", not general
 * C++ iterators. This hot-fix adds support for any arbitrary iterator type.
 */
template <typename It>
MGPU_DEVICE typename std::iterator_traits<It>::value_type ldg(It ptr) {
    return *ptr;
}

};
