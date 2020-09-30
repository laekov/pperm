cmake_minimum_required(VERSION 3.2)

project(PPerm)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fopenmp -O2 -std=c++17")

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc	
)

add_executable(pperm ${SRCS})


