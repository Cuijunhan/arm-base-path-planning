cmake_minimum_required(VERSION 3.10)
project(move_demo)
set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

add_executable(move_demo main.cpp)

target_include_directories(move_demo PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(move_demo PRIVATE
    ${Python3_LIBRARIES}
)