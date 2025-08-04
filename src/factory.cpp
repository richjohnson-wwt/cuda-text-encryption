
#include "cpu_encryptor.hpp"
#ifdef USE_CUDA
#include "gpu_encryptor.hpp"
#endif

#include <memory>



std::unique_ptr<Encryptor> create_encryptor(bool use_cuda) {
#ifdef USE_CUDA
    if (use_cuda) {
        return std::make_unique<GPUEncryptor>();
    }
#endif
    return std::make_unique<CPUEncryptor>();
}
    