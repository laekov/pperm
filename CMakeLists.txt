cmake_minimum_required(VERSION 3.8)

project(PPerm LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp -O2 -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O2 -std=c++14")

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
	find_package("CUDAToolkit" 10)
else()
	find_package("CUDA" 10)
endif()

find_package(MPI)

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc
)
set_source_files_properties(${SRCS} PROPERTIES LANGUAGE CXX)

if("${CUDAToolkit_FOUND}" STREQUAL "TRUE" OR "${CUDA_FOUND}" STREQUAL "TRUE")
	enable_language(CUDA)
	file(GLOB_RECURSE SRCS_CUDA
		src/*.cu
	)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPPERM_CUDA")
endif()


add_executable(pperm ${SRCS} ${SRCS_CUDA})

if(${MPI_FOUND})
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})
	target_link_libraries(pperm ${MPI_CXX_LIBRARIES})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPPERM_MPI")
endif()
