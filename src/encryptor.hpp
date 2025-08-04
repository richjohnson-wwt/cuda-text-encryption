#pragma once
#include <vector>
#include <string>

class Encryptor {
public:
    virtual ~Encryptor() = default;
    virtual void encrypt_lines(std::vector<std::string>& lines) = 0;
    virtual void decrypt_lines(std::vector<std::string>& lines) = 0;
};
