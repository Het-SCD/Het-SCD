#pragma once

#include <moderngpu.cuh>

namespace mgpu {

template <typename T>
MGPU_DEVICE typename std::iterator_traits<T>::value_type ldg(T ptr) {
    return *ptr;
}

};
