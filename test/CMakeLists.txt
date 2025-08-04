
find_package(Catch2 REQUIRED)

if (USE_CUDA)
    add_executable(test_encrypt test_encrypt.cpp ../src/cpu_encryptor.cpp ../src/factory.cpp ../src/gpu_encryptor.cu)
else()
    add_executable(test_encrypt test_encrypt.cpp ../src/cpu_encryptor.cpp ../src/factory.cpp)
endif()

if(USE_CUDA)
    set_property(TARGET test_encrypt PROPERTY CUDA_ARCHITECTURES 89)
    target_link_libraries(test_encrypt PRIVATE Catch2::Catch2WithMain CUDA::cudart)
else()
    target_link_libraries(test_encrypt Catch2::Catch2WithMain)
endif()

include(CTest)
include(Catch)
catch_discover_tests(test_encrypt)