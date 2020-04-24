cmake_minimum_required(VERSION 3.11)
project(tflite-opencv-example)

set(CMAKE_C_FLAGS "-Wall -pthread")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

# GET CPU Architechture
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    message("Building for x86_64 platform.")
    set(ARCH "k8")
    set(TF_COMMAND make -j8 BUILD_WITH_NNAPI=false -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile lib)
    set(TF_INSTALL_PREFIX "linux_x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("Building for aarch64 platform.")
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(TF_INSTALL_PREFIX "generic-aarch64_armv8-a")
    set(CROSS_PREFIX "aarch64-linux-gnu-")
    set(TF_COMMAND make TARGET=generic-aarch64 TARGET_ARCH=armv8-a -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
    message("Building for armv7l platform using the rpi script from tensorflow.")
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(TF_INSTALL_PREFIX "rpi_armv7l")
    set(CROSS_PREFIX "arm-linux-gnueabihf-")
    set(TF_COMMAND make -j4 TARGET=rpi TARGET_ARCH=armv7l -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
else()
    message(FATAL_ERROR "Not implemented to build for: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message("System ARCH is ${CMAKE_SYSTEM_PROCESSOR} ${ARCH}")

include(ExternalProject)

ExternalProject_Add(opencv
  GIT_REPOSITORY https://github.com/cedricve/opencv
  GIT_TAG "3.3.0"
  SOURCE_DIR opencv
  BINARY_DIR opencv-build
  PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/scripts/cv.patch
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=RELEASE 
    -DFORCE_VTK=ON
    -DWITH_TBB=ON
    -DWITH_V4L=ON
    -DWITH_QT=OFF
    -DWITH_GTK_2_x=ON
    -DWITH_OPENGL=ON
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/opencv_install
)
set(OPENCV_INCLUDE_DIR ${CMAKE_BINARY_DIR}/opencv_install/include/)
set(OPENCV_LIBRARY_DIR ${CMAKE_BINARY_DIR}/opencv_install/lib/)
set(OPENCV_LIBRARIES opencv_imgproc opencv_core opencv_highgui opencv_video opencv_videoio opencv_imgcodecs opencv_features2d)
if(EXISTS "${CMAKE_BINARY_DIR}/opencv_install/share/OpenCV/OpenCVConfig.cmake")
    include(${CMAKE_BINARY_DIR}/opencv_install/share/OpenCV/OpenCVConfig.cmake)
    add_custom_target(rerun)
else()
    add_custom_target(rerun ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS opencv)
endif()

ExternalProject_Add(absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_PROGRESS 1
    GIT_SHALLOW 1
    PREFIX "absl"
    PATCH_COMMAND sed -i /ABSL_RANDOM_HWAES_X64_FLAGS/d
    ${CMAKE_BINARY_DIR}/absl/src/absl/absl/copts/AbseilConfigureCopts.cmake
    && sed -i /random/d ${CMAKE_BINARY_DIR}/absl/src/absl/absl/CMakeLists.txt
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/absl_install -DCMAKE_INSTALL_LIBDIR=lib
    BUILD_BYPRODUCTS libabsl_synchronization.a libabsl_symbolize.a
)
set(ABSL_LIBS absl_synchronization absl_stacktrace absl_symbolize absl_demangle_internal absl_debugging_internal absl_dynamic_annotations absl_time absl_time_zone absl_graphcycles_internal absl_failure_signal_handler absl_malloc_internal absl_base absl_spinlock_wait)

ExternalProject_Add(glog
    GIT_REPOSITORY https://github.com/google/glog
    GIT_PROGRESS 1
    GIT_SHALLOW 1
    CMAKE_ARGS -DWITH_GFLAGS=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/glog_install
    -DCMAKE_INSTALL_LIBDIR=lib
    PREFIX "glog"
    BUILD_BYPRODUCTS libglog.a
)
set(GLOG_LIBS ${CMAKE_BINARY_DIR}/glog_install/lib/libglog.a)


# Must match with commit that libedgetpu.so was built from.
set(TENSORFLOW_COMMIT d855adfc5a0195788bf5f92c3c7352e638aa1109)
ExternalProject_Add(tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow
    GIT_TAG ${TENSORFLOW_COMMIT}
    PREFIX "tensorflow"
    CONFIGURE_COMMAND ./tensorflow/lite/tools/make/download_dependencies.sh
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${TF_COMMAND}
    BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
    BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
    INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/gen/${TF_INSTALL_PREFIX}/lib/libtensorflow-lite.a ${CMAKE_BINARY_DIR}/
)
set(TF_LITE_LIB "${CMAKE_BINARY_DIR}/libtensorflow-lite.a")

# Setting all include and link directories.
set(LIB_EDGETPU "${CMAKE_SOURCE_DIR}/libedgetpu/direct/${ARCH}/libedgetpu.so.1.0")
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/libedgetpu/)
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/include/thirdparty/cxxopts)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/absl/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/)
include_directories(${CMAKE_BINARY_DIR}/glog_install/include)
include_directories(${CMAKE_BINARY_DIR}/opencv_install/include)
link_directories(${CMAKE_BINARY_DIR})
link_directories(${CMAKE_BINARY_DIR}/absl_install/lib)
link_directories(${CMAKE_BINARY_DIR}/glog_install/lib)
link_directories(${CMAKE_BINARY_DIR}/opencv_install/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${ARCH})

add_library(label_utils
  src/label_utils.cc
  include/label_utils.h)

add_library(tflite_wrapper
  src/tflite_wrapper.cc
  include/tflite_wrapper.h)
target_link_libraries(tflite_wrapper label_utils ${TF_LITE_LIB})
add_dependencies(tflite_wrapper label_utils tensorflow)

add_library(detection_camera
  src/detection_camera.cc
  include/detection_camera.h)
target_link_libraries(detection_camera tflite_wrapper ${TF_LITE_LIB} ${OPENCV_LIBRARIES})
add_dependencies(detection_camera tflite_wrapper tensorflow opencv)

add_executable(edge
  src/main.cpp
  ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
  ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
)
target_link_libraries(edge detection_camera tflite_wrapper ${OPENCV_LIBRARIES} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(edge detection_camera tflite_wrapper tensorflow opencv)

