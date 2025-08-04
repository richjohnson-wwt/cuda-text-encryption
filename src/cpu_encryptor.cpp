#include "encryptor.hpp"
#include "cpu_encryptor.hpp"
#include <iostream>

namespace {
    constexpr char XOR_KEY = 0xAA;

    void xor_transform(std::vector<std::string>& lines) {
        for (auto& line : lines) {
            for (char& c : line) {
                c ^= XOR_KEY;
            }
        }
    }
}
    
void CPUEncryptor::encrypt_lines(std::vector<std::string>& lines) {
    xor_transform(lines);
}

void CPUEncryptor::decrypt_lines(std::vector<std::string>& lines) {
    xor_transform(lines);
}
