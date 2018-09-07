cmake_minimum_required(VERSION 2.8)
project(ldso)

# Set default build type if not specified otherwise.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 11)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# required libs
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Pangolin REQUIRED)

# optional libs
find_package(LibZip QUIET)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "${SSE_FLAGS} -O3 -Wno-deprecated -march=native -Wno-duplicate-decl-specifier -Wno-ignored-qualifiers -Wno-reorder -pthread -std=c++11")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
  message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
  add_definitions(-DHAS_ZIPLIB=1)
  include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
  message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
  set(LIBZIP_LIBRARY "")
endif()

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${Glog_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}

        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/thirdparty/
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/
)

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/build/src/libDBoW3.so
        ${LIBZIP_LIBRARY}
)

add_subdirectory(src)
add_subdirectory(examples)
