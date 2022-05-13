cmake_minimum_required(VERSION 2.8)
project(BlackHoleRayTracer CXX CUDA)
find_package(CUDA)
find_package(OpenCV)
add_subdirectory(pybind11)
##################
# Build on Linux #
##################
set(LINUX UNIX AND NOT APPLE)
if (LINUX)

    # GCC only
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

        set(GCC_CXX_FLAGS "-std=gnu++11 -m64")
        set(CMAKE_C_FLAGS fPIC)
        set(CMAKE_CXX_FLAGS -fPIC)
        # Debug configuration
        if (BUILD_DEBUG)
            set(CMAKE_BUILD_TYPE Debug)
            set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -g")
        else (BUILD_DEBUG)
            set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -O3")
            set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fopenmp")
        endif (BUILD_DEBUG)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS}")

    endif ()

endif (LINUX)


#-------------------------------------------------------------------------------
# Add subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(src)


# Install settings
set(CMAKE_INSTALL_PREFIX "${BlackHoleTracer_SOURCE_DIR}/")

add_library(RayTraceBlackHole MODULE Program.cpp)
set_target_properties(RayTraceBlackHole PROPERTIES
#        CXX_VISIBILITY_PRESET "hidden"
#        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )
target_include_directories(RayTraceBlackHole PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc .. ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(RayTraceBlackHole PUBLIC SchwarzschildRayProcessor)
target_link_libraries(RayTraceBlackHole PUBLIC argbcolor)
target_link_libraries(RayTraceBlackHole PUBLIC scene)
target_link_libraries(RayTraceBlackHole PUBLIC para)
target_link_libraries(RayTraceBlackHole PUBLIC ${OpenCV_LIBS})
target_link_libraries(RayTraceBlackHole PRIVATE pybind11::module)
target_link_libraries(RayTraceBlackHole PRIVATE ${CUDA_LIBRARIES})