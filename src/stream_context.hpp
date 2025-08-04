// stream_context.hpp
#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
using StreamHandle = cudaStream_t;
#else
// Dummy type for CPU build
using StreamHandle = void*;
#endif
