cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript1)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(torchscript1 classifier.cpp)
target_link_libraries(torchscript1 "${TORCH_LIBRARIES}")
target_link_libraries(torchscript1 ${OpenCV_LIBS})

set_property(TARGET torchscript1 PROPERTY CXX_STANDARD 11)

