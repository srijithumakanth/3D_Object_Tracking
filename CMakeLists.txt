cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

# Custom addition
find_package(Boost REQUIRED COMPONENTS system)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(PCL 1.2 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS} ${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES})