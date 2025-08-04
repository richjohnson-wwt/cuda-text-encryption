#pragma once
#include "encryptor.hpp"
#include <string>
#include <vector>

class CPUEncryptor : public Encryptor {
public:
    void encrypt_lines(std::vector<std::string>& lines) override;
    void decrypt_lines(std::vector<std::string>& lines) override;
};
