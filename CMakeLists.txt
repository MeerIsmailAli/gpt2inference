cmake_minimum_required(VERSION 3.16)

project(gpt2_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build tokenizers-cpp
add_subdirectory(
    dependencies/third_party/tokenizers-cpp
)

# Our executable
add_executable(gpt2 main.cpp)

# Link tokenizer library
target_link_libraries(gpt2 PRIVATE tokenizers_cpp)
