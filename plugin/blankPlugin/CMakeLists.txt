cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME mujoco_blank_plugin)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(${PROJECT_NAME} LANGUAGES CXX)

# should get MUJOCO_ROOT_DIR from command line
if(NOT DEFINED MUJOCO_ROOT_DIR)
    message(FATAL_ERROR "Set \"MUJOCO_ROOT_DIR\"")
elseif(NOT EXISTS ${MUJOCO_ROOT_DIR})
    message(FATAL_ERROR "\"MUJOCO_ROOT_DIR\" is invalid path: ${MUJOCO_ROOT_DIR}")
endif()

set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/include")
set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}")
set(MUJOCO_BIN_DIR "${MUJOCO_ROOT_DIR}/bin")

# find files that match these patterns and save to LIB_MUJOCO variable.
file(GLOB LIB_MUJOCO
    ${MUJOCO_BIN_DIR}/libmujoco[0-9][0-9][0-9].so
    ${MUJOCO_LIB_DIR}/libmujoco.so.*
)

if(EXISTS ${LIB_MUJOCO})
    message(STATUS "MuJoCo lib found: " ${LIB_MUJOCO})
else()
    message(FATAL_ERROR "MuJoCo lib not found.")
endif()

set(MUJOCO_BLANK_SRCS
    register.cpp
    blank_plugin.cpp
)

set(MUJOCO_BLANK_INCLUDES
    blank_plugin.hpp
)

set(MUJOCO_BLANK_INCLUDE
    ${MUJOCO_INCLUDE_DIR}
    # ${MUJOCO_BLANK_INCLUDES}
)

add_library(blank SHARED)
target_sources(blank PRIVATE ${MUJOCO_BLANK_SRCS})
target_include_directories(blank PRIVATE ${MUJOCO_BLANK_INCLUDE})
target_link_libraries(blank PRIVATE ${LIB_MUJOCO})

# target_compile_options(
#     blank
#     PRIVATE ${AVX_COMPILE_OPTIONS}
#     ${MUJOCO_MACOS_COMPILE_OPTIONS}
#     ${EXTRA_COMPILE_OPTIONS}
#     ${MUJOCO_CXX_FLAGS}
# )
# target_link_options(
#     blank
#     PRIVATE
#     ${MUJOCO_MACOS_LINK_OPTIONS}
#     ${EXTRA_LINK_OPTIONS}
# )