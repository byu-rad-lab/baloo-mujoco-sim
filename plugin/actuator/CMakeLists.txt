cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME mujoco_baloo_elevator_actuator_plugin)
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
  baloo_elevator.cpp
)

set(MUJOCO_BLANK_INCLUDES
  baloo_elevator.hpp
)

set(MUJOCO_BLANK_INCLUDE
  ${MUJOCO_INCLUDE_DIR}
)

add_library(baloo_elevator SHARED)
target_sources(baloo_elevator PRIVATE ${MUJOCO_BLANK_SRCS})
target_include_directories(baloo_elevator PRIVATE ${MUJOCO_BLANK_INCLUDE})
target_link_libraries(baloo_elevator PRIVATE ${LIB_MUJOCO})

install(TARGETS baloo_elevator
  LIBRARY DESTINATION ${MUJOCO_ROOT_DIR}/plugin)
