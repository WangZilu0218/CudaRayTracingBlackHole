cmake_minimum_required(VERSION 3.10)
#project(RayTraceBlackholeCuda CXX CUDA)
#find_package(CUDA 9.0 REQUIRED)

if(CUDA_FOUND)
    message(STATUS ${CUDA_INCLUDE_DIRS})
    message(STATUS ${PROJECT_SOURCE_DIR})
    message(STATUS ${CUDA_LIBRARIES})
endif()
find_package(OpenMP)
set(CMAKE_CXX_STANDARD 14)
if(OpenMP_FOUND)
    message(STATUS "OpenMP includes: ${OPENMP_INCLUDES}")
    message(STATUS "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_NVCC_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(OpenCV REQUIRED)
add_library(utils utiles/utils.cu)
target_include_directories(utils PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS})
set_target_properties(utils PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )

#include_directories(${OpenCV_INCLUDE_DIRS})
add_library(argbcolor utiles/ArgbColor.cu)
target_include_directories(argbcolor PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS})
set_target_properties(argbcolor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )

add_library(sky hitables/Sky.cu)
target_include_directories(sky PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
set_target_properties(sky PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )
target_link_libraries(sky PRIVATE utils)
add_library(horizon hitables/Horizon.cu)
target_include_directories(horizon PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(horizon PRIVATE utils)
set_target_properties(horizon PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )
add_library(textureDisk hitables/TexturedDisk.cu)
target_include_directories(textureDisk PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(textureDisk PRIVATE utils)
set_target_properties(textureDisk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )
add_library(SchwarzschildRayProcessor processor/SchwarzschildRayProcessor.cu processor/SchwarzschildRayProcessor.cpp)
target_include_directories(SchwarzschildRayProcessor PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(SchwarzschildRayProcessor PUBLIC utils sky textureDisk horizon)
set_target_properties(SchwarzschildRayProcessor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        #        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_VERBOSE_BUILD ON
        #        CUDA_VISIBILITY_PRESET "hidden"
        CUDA_ARCHITECTURES "75"
        CUDA_SEPARABLE_COMPILATION ON
        )

add_library(scene Scene.cpp)
target_include_directories(scene PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(scene PUBLIC ${CUDA_LIBRARIES} utils)

add_library(para utiles/parallelStaff.cpp)
target_include_directories(para PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)



