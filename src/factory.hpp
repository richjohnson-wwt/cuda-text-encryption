#include "encryptor.hpp"
#include <memory>

std::unique_ptr<Encryptor> create_encryptor(bool use_cuda);
