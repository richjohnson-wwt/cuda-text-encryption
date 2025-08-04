
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/cpu_encryptor.hpp" 

TEST_CASE("CPUEncryptor encrypts and decrypts lines", "[encryption]") {
    CPUEncryptor enc;
    std::vector<std::string> lines = { "Call me Ishmael.", "Some years ago..." };

    enc.encrypt_lines(lines);
    REQUIRE(lines[0] != "Call me Ishmael.");

    enc.decrypt_lines(lines);
    REQUIRE(lines[0] == "Call me Ishmael.");
    REQUIRE(lines[1] == "Some years ago...");
}

#ifdef USE_CUDA
TEST_CASE("GPUEncryptor encrypts/decrypts") {
    GPUEncryptor enc;
    std::vector<std::string> lines = { "Call me Ishmael.", "Some years ago..."   };
    enc.encrypt_lines(lines);
    enc.decrypt_lines(lines);
    REQUIRE(lines[0] == "Call me Ishmael.");
    REQUIRE(lines[1] == "Some years ago...");
}
#endif
