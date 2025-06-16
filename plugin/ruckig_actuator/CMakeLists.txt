cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME mujoco_ruckig_actuator_plugin)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(${PROJECT_NAME} LANGUAGES CXX)

find_package(ruckig REQUIRED)

# should get MUJOCO_ROOT_DIR from command line
if(NOT DEFINED MUJOCO_ROOT_DIR)
  message(FATAL_ERROR "Set \"MUJOCO_ROOT_DIR\"")
elseif(NOT EXISTS ${MUJOCO_ROOT_DIR})
  message(FATAL_ERROR "\"MUJOCO_ROOT_DIR\" is invalid path: ${MUJOCO_ROOT_DIR}")
endif()

set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/include")
set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}/lib")
set(MUJOCO_BIN_DIR "${MUJOCO_ROOT_DIR}/bin")
set(MUJOCO_PLUGIN_DIR "${MUJOCO_ROOT_DIR}/plugin")

# find files that match these patterns and save to LIB_MUJOCO variable.
message(STATUS "Looking for MuJoCo lib in " ${MUJOCO_LIB_DIR}, ${MUJOCO_ROOT_DIR} " and " ${MUJOCO_BIN_DIR} "...")
file(GLOB LIB_MUJOCO
  ${MUJOCO_BIN_DIR}/libmujoco.so.[0-9].[0-9].[0-9]
  ${MUJOCO_LIB_DIR}/libmujoco.so.[0-9].[0-9].[0-9] # typical of binary installations
  ${MUJOCO_ROOT_DIR}/libmujoco.so.[0-9].[0-9].[0-9] # typical of pip installations
)

if(EXISTS ${LIB_MUJOCO})
  message(STATUS "MuJoCo lib found: " ${LIB_MUJOCO})
else()
  message(FATAL_ERROR "MuJoCo lib not found.")
endif()

set(MUJOCO_ACTUATOR_SRCS
  register.cpp
  RuckigActuator.cpp
)

set(MUJOCO_ACTUATOR_INCLUDES
  RuckigActuator.hpp
)

set(MUJOCO_ACTUATOR_INCLUDE
  ${MUJOCO_INCLUDE_DIR}
)

add_library(RuckigActuator SHARED)
target_sources(RuckigActuator PRIVATE ${MUJOCO_ACTUATOR_SRCS})
target_include_directories(RuckigActuator PRIVATE ${MUJOCO_ACTUATOR_INCLUDE})
target_link_libraries(RuckigActuator PRIVATE ${LIB_MUJOCO} ruckig::ruckig)

install(
  TARGETS RuckigActuator LIBRARY
  DESTINATION ${MUJOCO_PLUGIN_DIR}
)

# optional argument to also install plugins to a precompiled MuJoCo directory
if(DEFINED PRECOMPILED_MUJOCO_DIR)
  message(STATUS "Also installing to precompiled MuJoCo directory: " ${PRECOMPILED_MUJOCO_DIR})
  install(
    TARGETS RuckigActuator LIBRARY
    DESTINATION ${PRECOMPILED_MUJOCO_DIR}/bin/mujoco_plugin
  )
endif()
