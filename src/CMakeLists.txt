if(USE_CUDA)
    set(SOURCES main.cpp gpu_encryptor.cu factory.cpp)
else()
    set(SOURCES main.cpp cpu_encryptor.cpp factory.cpp)
endif()

add_executable(moby-dick ${SOURCES})

if(USE_CUDA)
    set_property(TARGET moby-dick PROPERTY CUDA_ARCHITECTURES 89)
endif()
