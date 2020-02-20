cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(project_name)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(project_name classifier.cpp)
target_link_libraries(project_name "${TORCH_LIBRARIES}")
target_link_libraries(project_name ${OpenCV_LIBS})

set_property(TARGET project_name PROPERTY CXX_STANDARD 11)

