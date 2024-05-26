cmake_minimum_required(VERSION 3.22.1)

enable_language(C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project("llama-mobile")

include(FetchContent)

FetchContent_Declare(
    spdlog
    URL "https://github.com/gabime/spdlog/archive/refs/tags/v1.14.0.zip"
)

FetchContent_Declare(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.3.283
)
FetchContent_Populate(vulkan_headers)
set(Vulkan_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include")

FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG        v0.9.0
)
set(KOMPUTE_OPT_USE_SPDLOG OFF)
set(KOMPUTE_OPT_LOG_LEVEL "Off")

FetchContent_Declare(
    llamacpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b2996
)

FetchContent_MakeAvailable(kompute spdlog llamacpp)

list(APPEND CMAKE_PREFIX_PATH "${kompute_SOURCE_DIR}/cmake")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE kompute spdlog::spdlog llama ggml)
