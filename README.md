### Uses XOR encryption to encrypt/decrypt text in parallel on GPU

### Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson

### Debug Config

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
    cd build/Debug 
    
    # All commands in build/Debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
    cmake --build .

    ./src/moby-dick --encrypt ../../data/chapter1.txt --use-cuda
    ./src/moby-dick --decrypt ../../data/chapter1.txt.enc --use-cuda

    ./src/moby-dick --encrypt ../../data/MobyDick.txt --use-cuda
    ./src/moby-dick --decrypt ../../data/MobyDick.txt.enc --use-cuda

#### On Macbook

    ./src/moby-dick --encrypt ../../data/MobyDick.txt
    Output written to: ../../data/MobyDick.txt.enc
    ✅ Output written to: ../../data/MobyDick.txt.enc
    📊 Timing stats:
    Read file:      30 ms
    Process lines:  5 ms
    Write file:     2 ms
    Total runtime:  38 ms
    
    (cuda-moby-dick) johnsori@D7D6690221 Debug % ./src/moby-dick --decrypt ../../data/MobyDick.txt.enc
    Output written to: ../../data/MobyDick.txt
    ✅ Output written to: ../../data/MobyDick.txt
    📊 Timing stats:
    Read file:      33 ms
    Process lines:  6 ms
    Write file:     2 ms
    Total runtime:  41 ms

#### On 5080 Ti

    (cuda-text-encryption) (main) root@C.24692078:/workspace/cuda-text-encryption/build/Debug$ ./src/moby-dick --encrypt ../../data/chapter1.txt --use-cuda
    [factory] CUDA support compiled in.
    Output written to: ../../data/chapter1.txt.enc
    ✅ Output written to: ../../data/chapter1.txt.enc
    📊 Timing stats:
    Read file:      0 ms
    Process lines:  2 ms
    Write file:     0 ms
    Total runtime:  183 ms

    (cuda-text-encryption) (main) root@C.24692078:/workspace/cuda-text-encryption/build/Debug$ ./src/moby-dick --decrypt ../../data/MobyDick.txt.enc --use-cuda
    [factory] CUDA support compiled in.
    Output written to: ../../data/MobyDick.txt
    ✅ Output written to: ../../data/MobyDick.txt
    📊 Timing stats:
    Read file:      1 ms
    Process lines:  4 ms
    Write file:     1 ms
    Total runtime:  190 ms
