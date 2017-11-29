project(MPC)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11 -O3)

set(CXX_FLAGS "-Wall ") #-undefined dynamic_lookup
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/MPC.cpp src/main.cpp)

include_directories(/usr/local/include)
include_directories(/usr/local/include/google/tensorflow)
include_directories(src/Eigen-3.3)
include_directories(src)

link_directories(/usr/local/lib)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  include_directories(/usr/local/opt/openssl/include)
  link_directories(/usr/local/opt/openssl/lib)
  link_directories(/usr/local/Cellar/libuv/1.11.0/lib)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(mpc ${sources})

target_link_libraries(mpc ipopt z ssl uv uWS tensorflow_cc)

