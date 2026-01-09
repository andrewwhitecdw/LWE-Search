cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")

project(SEALExamples VERSION 4.1.1 LANGUAGES CXX CUDA)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit version ${CUDAToolkit_VERSION}")

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 90 CACHE STRING "CUDA architectures" FORCE)
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "--forward-unknown-to-host-compiler -Wno-shadow --expt-relaxed-constexpr --extended-lambda")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG")
endif()

# Find CUDA driver library
find_library(CUDA_DRIVER_LIBRARY NAMES cuda PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "CUDA driver library (libcuda.so) not found!")
endif()
message(STATUS "Found CUDA driver library: ${CUDA_DRIVER_LIBRARY}")

# Use CUDASTF from the CCCL project
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY caugonnet/cccl
    GIT_TAG stf_stackable_cybertron
    OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Fetch CUTLASS
CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY nvidia/cutlass
    GIT_TAG v4.1.0
    OPTIONS
        "CUTLASS_ENABLE_HEADERS_ONLY ON"
        "CUTLASS_ENABLE_EXAMPLES OFF"
        "CUTLASS_ENABLE_TESTS OFF"
        "CUTLASS_ENABLE_TOOLS OFF"
)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: found")
endif()

# Fetch Microsoft SEAL using CPM
CPMAddPackage(
    NAME SEAL
    GITHUB_REPOSITORY microsoft/SEAL
    GIT_TAG v4.1.1
    OPTIONS
        "SEAL_USE_MSGSL ON"
        "SEAL_USE_ZLIB OFF"
        "SEAL_USE_ZSTD OFF"
        "SEAL_BUILD_EXAMPLES OFF"
        "SEAL_BUILD_TESTS OFF"
        "SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT OFF"
)

# Apply patch to fix primitive root generation for 0x7FFFFFE060001
if(SEAL_ADDED)
    set(SEAL_NUMTH_FILE "${SEAL_SOURCE_DIR}/native/src/seal/util/numth.cpp")
    set(SEAL_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/numth-primitive-root.patch")
    
    # Check if patch is already applied
    file(READ "${SEAL_NUMTH_FILE}" NUMTH_CONTENT)
    string(FIND "${NUMTH_CONTENT}" "bool special = (p == 0x7FFFFFE060001ULL)" PATCH_FOUND)
    
    if(PATCH_FOUND EQUAL -1)
        message(STATUS "Applying SEAL patch: ${SEAL_PATCH_FILE}")
        execute_process(
            COMMAND patch -p1 -i "${SEAL_PATCH_FILE}"
            WORKING_DIRECTORY "${SEAL_SOURCE_DIR}"
            RESULT_VARIABLE PATCH_RESULT
            OUTPUT_VARIABLE PATCH_OUTPUT
            ERROR_VARIABLE PATCH_ERROR
        )
        if(NOT PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply SEAL patch:\n${PATCH_OUTPUT}\n${PATCH_ERROR}")
        else()
            message(STATUS "SEAL patch applied successfully")
        endif()
    else()
        message(STATUS "SEAL patch already applied")
    endif()
endif()

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

include_directories(
    ${cutlass_SOURCE_DIR}/include/
)

add_library(pir_kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lwe_server.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rlwe_server.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simplepir.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rlwe_compress.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rlwe_kernels/ntt_twiddles.cu
)

target_include_directories(pir_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rlwe_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${cutlass_SOURCE_DIR}/include
)

target_link_libraries(pir_kernels PUBLIC
    CCCL::CCCL
    CCCL::cudax
    CUDA::cudart
    ${CUDA_DRIVER_LIBRARY}
    SEAL::seal
)

set_target_properties(pir_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
)

set(EXAMPLES
    LWE_TEST
    RLWE_TEST
    PIR
    PIR_3D
)

foreach(example IN LISTS EXAMPLES)
    add_executable(${example} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}.cpp)
    
    set_target_properties(${example} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
    )
    
    target_include_directories(${example} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${CMAKE_CURRENT_SOURCE_DIR}/rlwe_kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${cutlass_SOURCE_DIR}/include
    )
    
    target_link_libraries(${example} PRIVATE
        SEAL::seal
        pir_kernels
        OpenMP::OpenMP_CXX
    )
endforeach()