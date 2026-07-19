cmake_minimum_required(VERSION 3.26)
project(tmopt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
# B300 = Blackwell Ultra. 12.8 toolkit compiles sm_100 (runs via forward
# compat); switch to 103 once the 12.9+/13.x toolchain lands (PLAN Stage 1B).
set(CMAKE_CUDA_ARCHITECTURES 100)

# single external dep: header-only JSON for config verification (startup-only)
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(inkling_serve
    src/main.cpp
    src/model_loader.cpp
    src/kv_cache.cpp
    src/scheduler.cpp
    kernels/moe_grouped_gemm.cu
    kernels/decode_attention.cu
    kernels/fused_rmsnorm.cu
    kernels/sconv.cu
)
target_include_directories(inkling_serve PRIVATE include)
target_link_libraries(inkling_serve PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(inkling_serve PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)

# config verification is testable without GPUs or weights beyond the JSONs:
add_executable(verify_config src/model_loader.cpp src/verify_config_main.cpp)
target_include_directories(verify_config PRIVATE include)
target_link_libraries(verify_config PRIVATE nlohmann_json::nlohmann_json)
