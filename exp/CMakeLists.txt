cmake_minimum_required(VERSION 3.10)
project(RegexANNSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address")

include_directories(include)

# All shared library sources (everything except *_main.cpp and main.cpp)
set(LIB_SRCS
    src/Baseline.cpp
    src/Eval.cpp
    src/GraphIndex.cpp
    src/HierarchicalIndex.cpp
    src/Index.cpp
    src/KMeans.cpp
    src/PQ.cpp
    src/RegexParser.cpp
    src/Search.cpp
    src/Serialize.cpp
)

add_library(regann_lib STATIC ${LIB_SRCS})
target_compile_options(regann_lib PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Main search binary
add_executable(regann src/main.cpp)
target_link_libraries(regann regann_lib)
target_compile_options(regann PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Standalone recall evaluation binary
add_executable(eval_recall src/eval_main.cpp)
target_link_libraries(eval_recall regann_lib)
target_compile_options(eval_recall PRIVATE -Wall -Wextra -Wno-unused-parameter)

install(TARGETS regann eval_recall DESTINATION bin)
