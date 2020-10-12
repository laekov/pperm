cmake_minimum_required(VERSION 3.8)

project(PPerm LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fopenmp -O2 -std=c++17")

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
	find_package("CUDAToolkit" 10)
else()
	find_package("CUDA" 10)
endif()

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc
)

if(CUDAToolkit_FOUND OR CUDA_FOUND)
	file(GLOB_RECURSE SRCS
		src/*.cu
	)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPPERM_CUDA")
endif()

add_executable(pperm ${SRCS})


