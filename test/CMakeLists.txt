
find_package(Catch2 REQUIRED)

add_executable(test_encrypt test_encrypt.cpp ../src/cpu_encryptor.cpp ../src/factory.cpp)
target_link_libraries(test_encrypt Catch2::Catch2WithMain)

if(USE_CUDA)
    set_property(TARGET test_encrypt PROPERTY CUDA_ARCHITECTURES 89)
endif()

include(CTest)
include(Catch)
catch_discover_tests(test_encrypt)