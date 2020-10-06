cmake_minimum_required(VERSION 3.8)

project(PPerm LANGUAGES CXX CUDA)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fopenmp -O2 -std=c++17")

find_package(CUDA 10)

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc	
	src/*.cu
)

add_executable(pperm ${SRCS})


