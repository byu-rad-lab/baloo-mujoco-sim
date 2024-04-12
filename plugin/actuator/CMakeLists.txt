# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME mujoco_actuator_plugin)
set(CMAKE_COLOR_DIAGNOSTICS ON)
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# should get MUJOCO_ROOT_DIR from command line
if(NOT DEFINED MUJOCO_ROOT_DIR)
  message(FATAL_ERROR "Set \"MUJOCO_ROOT_DIR\"")
elseif(NOT EXISTS ${MUJOCO_ROOT_DIR})
  message(FATAL_ERROR "\"MUJOCO_ROOT_DIR\" is invalid path: ${MUJOCO_ROOT_DIR}")
endif()

# Define an option that the user can set
option(MUJOCO_INSTALLED_VIA_PIP "Was MuJoCo installed via pip?" OFF)

# Set MUJOCO_INCLUDE_DIR based on the value of MUJOCO_INSTALLED_VIA_PIP
if(MUJOCO_INSTALLED_VIA_PIP)
  set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/include")
  set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}")
else()
  set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/../include")
  message(STATUS "MUJOCO_INCLUDE_DIR: ${MUJOCO_INCLUDE_DIR}")
  set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}")
  set(MUJOCO_BIN_DIR "${MUJOCO_ROOT_DIR}/../bin")
endif()


# set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/include")
# set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}")
# set(MUJOCO_BIN_DIR "${MUJOCO_ROOT_DIR}/bin")

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

# set(MUJOCO_ACTUATOR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..
#                             ${CMAKE_CURRENT_SOURCE_DIR}/../../src
# )

set(MUJOCO_ACTUATOR_INCLUDE
    ${MUJOCO_INCLUDE_DIR}
)

set(MUJOCO_ACTUATOR_SRCS
    quintic.cc
    quintic.h
    register.cc
)

add_library(actuator SHARED)
target_sources(actuator PRIVATE ${MUJOCO_ACTUATOR_SRCS})
target_include_directories(actuator PRIVATE ${MUJOCO_ACTUATOR_INCLUDE})
target_link_libraries(actuator PRIVATE ${LIB_MUJOCO})

target_compile_options(
  actuator
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)

target_link_options(
  actuator
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)
