cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME mujoco_joint_angle_estimator_plugin)
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
set(MUJOCO_PLUGIN_DIR "${MUJOCO_ROOT_DIR}/plugin")

# find files that match these patterns and save to LIB_MUJOCO variable.
message(STATUS "Looking for MuJoCo lib in " ${MUJOCO_LIB_DIR} " and " ${MUJOCO_BIN_DIR} "...")
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
    joint_angle_estimator.cpp
)

set(MUJOCO_BLANK_INCLUDES
    joint_angle_estimator.hpp
)

set(MUJOCO_BLANK_INCLUDE
    ${MUJOCO_INCLUDE_DIR}
)

add_library(joint_angle_estimator SHARED)
target_sources(joint_angle_estimator PRIVATE ${MUJOCO_BLANK_SRCS})
target_include_directories(joint_angle_estimator PRIVATE ${MUJOCO_BLANK_INCLUDE})
target_link_libraries(joint_angle_estimator PRIVATE ${LIB_MUJOCO})

# check if being built with SKBUILD or with CMake
if(SKBUILD)
    message(STATUS "Building with SKBUILD")
    install(TARGETS joint_angle_estimator
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

    # then copy files over outside
    message(STATUS "Copying ${CMAKE_BINARY_DIR}/joint_angle_estimator/libjoint_angle_estimator.so to ${MUJOCO_PLUGIN_DIR}")
    file(
        COPY ${CMAKE_BINARY_DIR}/joint_angle_estimator/libjoint_angle_estimator.so
        DESTINATION ${MUJOCO_PLUGIN_DIR}
    )
else()
    message(STATUS "Building with CMake")
    install(TARGETS joint_angle_estimator
        LIBRARY DESTINATION ${MUJOCO_PLUGIN_DIR})
endif()

# message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_CURRENT_BINARY_DIR})
