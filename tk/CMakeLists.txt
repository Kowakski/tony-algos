# cmake needs this line
# g++ `pkg-config opencv --cflags` example_04-01.cpp -o 0401_exe `pkg-config opencv --libs`
cmake_minimum_required(VERSION 3.5)

# Define project name
project(opencvTest)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
# set(OpenCV_DIR /opt/opencv-2.4.11/share/OpenCV) 指定版本
find_package(OpenCV REQUIRED)

#option(EXP "enable example compilation" OFF)
option(EXP "enable debug compilation" OFF)

add_compile_options(-std=c++11 -Wall -g)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(main ./src/main.cpp)

# Declare the executable target built from your sources
if(EXP)
    add_executable(kalman ./src/kalman_demo.cpp)
    target_link_libraries(kalman ${OpenCV_LIBS})
else()
    #message( STATUS "exp codes not open" )
    add_executable(main ./src/main.cpp)
    target_link_libraries(main ${OpenCV_LIBS})
endif()

# Link your application with OpenCV libraries
# target_link_libraries(main ${OpenCV_LIBS})
