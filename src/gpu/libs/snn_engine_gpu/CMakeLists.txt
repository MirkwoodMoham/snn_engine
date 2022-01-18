cmake_minimum_required(VERSION 3.6)

project(snn_engine_gpu)

enable_language(CUDA)
find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()


find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)
find_package(OpenGL REQUIRED)

get_filename_component(PROJECT_INC_DIRECTORY ../../../../inc ABSOLUTE)

include_directories(
    ./src/
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_INC_DIRECTORY}
    ${OPENGL_INCLUDE_DIR}

)

link_directories(
    /usr/local/cuda/lib64
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/libs/x64"
)


if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


cuda_add_library(
    snn_engine_gpu SHARED 
    ./src/bindings.cu 
    ./src/snn_construction.cu
    ./src/example/snn_engine_gpu.cu
    ./src/utils/launch_parameters.cu
)

target_link_libraries(
    snn_engine_gpu
    ${PYTHON_LIBRARIES}
    cudart
)

set_target_properties(snn_engine_gpu PROPERTIES PREFIX "")

IF (WIN32)
    set_target_properties(snn_engine_gpu PROPERTIES SUFFIX ".pyd")
ENDIF()