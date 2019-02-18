set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( sift )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} src/features_compute/include)

add_subdirectory(src/main)
add_subdirectory(src/features_compute)