### 根目录下 CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(RegexANNSearch)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(regann ${SOURCES})