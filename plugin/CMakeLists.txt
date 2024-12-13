set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(baloo_mujoco_plugins)

# Add the subdirectories
add_subdirectory(joint_angle_estimator)
add_subdirectory(motion_profile_servo)