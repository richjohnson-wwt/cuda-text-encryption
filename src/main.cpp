#include "encryptor.hpp"
#include "factory.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

void read_lines(const std::string& filename, std::vector<std::string>& lines) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Failed to open file");
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
}

void write_lines(const std::string& filename, const std::vector<std::string>& lines) {
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Failed to open file");
    for (const auto& line : lines) {
        output << line << "\n";
    }
}

std::string output_filename(const std::string& input, bool encrypting) {
    if (encrypting) {
        return input + ".enc";
    } else {
        if (input.size() >= 4 && input.substr(input.size() - 4) == ".enc") {
            return input.substr(0, input.size() - 4); // remove ".enc"
        } else {
            throw std::runtime_error("Decrypt input file must have .enc extension");
        }
    }
}

int main(int argc, char* argv[]) {
    bool use_cuda = false;
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [--encrypt|--decrypt] <filename>\n";
        return 1;
    }
    if (argc == 4 && std::string(argv[3]) == "--use-cuda") {
        use_cuda = true;
    }

    std::string mode = argv[1];
    std::string input_filename = argv[2];
    std::vector<std::string> lines;

    try {
        using Clock = std::chrono::high_resolution_clock;
        auto t_start = Clock::now();
        auto t_read_start = Clock::now();
        read_lines(input_filename, lines);
        auto t_read_end = Clock::now();

        std::unique_ptr<Encryptor> encryptor = create_encryptor(use_cuda);
        auto t_process_start = Clock::now();
        if (mode == "--encrypt") {
            encryptor->encrypt_lines(lines);
        } else if (mode == "--decrypt") {
            encryptor->decrypt_lines(lines);
        } else {
            std::cerr << "Invalid mode: " << mode << "\n";
            return 1;
        }
        auto t_process_end = Clock::now();

        std::string output = output_filename(input_filename, mode == "--encrypt");
        auto t_write_start = Clock::now();
        write_lines(output, lines);
        auto t_write_end = Clock::now();
        std::cout << "Output written to: " << output << "\n";
        auto t_end = Clock::now();
        // Print stats
        auto ms = [](auto start, auto end) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        };

        std::cout << "✅ Output written to: " << output << "\n";
        std::cout << "📊 Timing stats:\n";
        std::cout << "  Read file:      " << ms(t_read_start, t_read_end) << " ms\n";
        std::cout << "  Process lines:  " << ms(t_process_start, t_process_end) << " ms\n";
        std::cout << "  Write file:     " << ms(t_write_start, t_write_end) << " ms\n";
        std::cout << "  Total runtime:  " << ms(t_start, t_end) << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
