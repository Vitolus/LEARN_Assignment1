cmake_minimum_required(VERSION 3.28)
project(LEARN_Assignment1)

set(CMAKE_CXX_STANDARD 20)

# Find OpenMP
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(LEARN_Assignment1 main.cpp
        page_rank.cpp
        page_rank.h)

# Link against OpenMP
if(OPENMP_FOUND)
    target_link_libraries(LEARN_Assignment1 PUBLIC OpenMP::OpenMP_CXX)
endif()
