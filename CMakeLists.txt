cmake_minimum_required(VERSION 3.10)
project(moby-dick VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    set(CMAKE_CUDA_ARCHITECTURES 89)
    message(STATUS "CUDA enabled")
else()
    message(STATUS "CUDA disabled — using CPU")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)


option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()