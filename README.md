# Baloo Simulation

This repository contains the code for a simulation of a robot named Baloo. The simulation is written in Python and uses the `dm_control` and `mujoco` libraries.

<!-- add  scresnshot.png-->
![Sim](screenshot.png)

## Overview

The main class in the code is `Baloo`, which represents the robot. The `Baloo` class has methods for setting up the simulation, including setting compiler options, visual settings, and contact settings. It also has methods for creating the robot's body parts and actuators.

The robot is composed of a number of disks, which can be specified when creating a `Baloo` instance. The robot also has a number of joints, and the height of these joints can be adjusted.

The simulation includes a world plane and a fixed camera view. There is also a box object in the simulation that the robot can interact with.

## Dependencies

To run the simulation, you will need Python and the `dm_control` and `mujoco` libraries. You can install these libraries using pip:

``` bash
pip install dm_control mujoco
```

## Installation and Setup

There are a few steps to install and set up the simulation:

1. Clone the repository
2. Navigate to the plugin directory and build/install the joint_angle_estimator and motion_profile_servo plugins. There is a top-level CMakeLists.txt file that will build and install all the plugins needed.
3. Install the simulation package locally using pip. This will set up everything to be nicely importable.
4. Run the generate_baloo_xml.py script to generate the xml file for the robot.

Here's an example of how to do this:

``` bash
git clone <repo-url>
cd baloo_mujoco_sim

# Build and install the plugins
cd plugins
mkdir build
cd build
cmake ..
make install

# Install the simulation package locally
cd ../..
pip install -e .

# Generate the xml file for the robot
cd ../../..
python generate_baloo_xml.py

```

## Usage
Once everything is built and installed, you can run a simulation as shown in the examples directory. 



