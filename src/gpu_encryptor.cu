#include "gpu_encryptor.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    __global__ void xor_kernel(char* data, int* line_offsets, int* line_lengths, int total_lines, char key) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_lines) {
            int offset = line_offsets[idx];
            int length = line_lengths[idx];
            for (int i = 0; i < length; ++i) {
                data[offset + i] ^= key;
            }
        }
    }
}

GPUEncryptor::GPUEncryptor() {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}

GPUEncryptor::~GPUEncryptor() {
    cudaStreamDestroy(stream_);
}

void GPUEncryptor::encrypt_lines(std::vector<std::string>& lines) {
    const char key = 0xAA;

    // Concatenate all lines into one buffer
    std::string flat;
    std::vector<int> line_offsets, line_lengths;
    int offset = 0;
    for (const auto& line : lines) {
        line_offsets.push_back(offset);
        line_lengths.push_back(static_cast<int>(line.size()));
        flat += line;
        offset += line.size();
    }

    int total_lines = static_cast<int>(lines.size());
    size_t total_bytes = flat.size();

    // Allocate device memory
    char* d_data = nullptr;
    int* d_offsets = nullptr;
    int* d_lengths = nullptr;

    cudaMallocAsync(&d_data, total_bytes, stream_);
    cudaMallocAsync(&d_offsets, sizeof(int) * total_lines, stream_);
    cudaMallocAsync(&d_lengths, sizeof(int) * total_lines, stream_);

    // Copy to device
    cudaMemcpyAsync(d_data, flat.data(), total_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_offsets, line_offsets.data(), sizeof(int) * total_lines, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_lengths, line_lengths.data(), sizeof(int) * total_lines, cudaMemcpyHostToDevice, stream_);

    // Launch kernel
    int threads_per_block = 128;
    int blocks = (total_lines + threads_per_block - 1) / threads_per_block;
    xor_kernel<<<blocks, threads_per_block, 0, stream_>>>(d_data, d_offsets, d_lengths, total_lines, key);

    // Copy back
    std::vector<char> result(total_bytes);
    cudaMemcpyAsync(result.data(), d_data, total_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Unflatten back into lines
    for (int i = 0; i < total_lines; ++i) {
        lines[i] = std::string(result.data() + line_offsets[i], line_lengths[i]);
    }

    // Free device memory
    cudaFreeAsync(d_data, stream_);
    cudaFreeAsync(d_offsets, stream_);
    cudaFreeAsync(d_lengths, stream_);
}

void GPUEncryptor::decrypt_lines(std::vector<std::string>& lines) {
    encrypt_lines(lines);  // Symmetric XOR
}
