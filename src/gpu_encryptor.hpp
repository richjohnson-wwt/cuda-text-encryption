#pragma once
#include "gpu_encryptor.hpp"
#include <string>
#include <vector>
#include <cuda_runtime.h>

class GPUEncryptor : public Encryptor {
public:
    GPUEncryptor();
    ~GPUEncryptor();

    void encrypt_lines(std::vector<std::string>& lines) override;
    void decrypt_lines(std::vector<std::string>& lines) override;

private:
    cudaStream_t stream_; // Only defined when USE_CUDA
};
