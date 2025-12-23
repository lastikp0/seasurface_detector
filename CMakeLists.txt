cmake_minimum_required(VERSION 3.16)
project(seasurface_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Root directory of ONNX Runtime")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    "${ONNXRUNTIME_ROOT}/include"
  PATH_SUFFIXES onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib64"
)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "Expected headers in: ${ONNXRUNTIME_ROOT}/include\n"
    "Expected libs in: ${ONNXRUNTIME_ROOT}/lib or lib64\n"
  )
endif()

get_filename_component(ORT_LIBDIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)

add_executable(seasurface_detector
  src/main.cpp
  src/detector.cpp
  src/tracker.cpp
  src/evaluator.cpp
  src/csv.cpp
  src/utils.cpp
)

target_include_directories(seasurface_detector PRIVATE
  src
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(seasurface_detector PRIVATE
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
  dl
)

set_target_properties(seasurface_detector PROPERTIES
  BUILD_RPATH "${ORT_LIBDIR}"
  INSTALL_RPATH "${ORT_LIBDIR}"
)

if (MSVC)
  target_compile_options(seasurface_detector PRIVATE /W4)
else()
  target_compile_options(seasurface_detector PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "ONNX Runtime libdir: ${ORT_LIBDIR}")
