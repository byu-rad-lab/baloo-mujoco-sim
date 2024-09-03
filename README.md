# Baloo Simulation

This repository contains the code for a simulation of a robot named Baloo. The simulation is written in Python and uses the `dm_control` and `mujoco` libraries.

<!-- add  scresnshot.png-->
![Sim](screenshot.png)

## Overview

The main class in the code is `Baloo`, which represents the robot. The `Baloo` class has methods for setting up the simulation, including setting compiler options, visual settings, and contact settings. It also has methods for creating the robot's body parts and actuators.

The robot is composed of a number of disks, which can be specified when creating a `Baloo` instance. The robot also has a number of joints, and the height of these joints can be adjusted.

The simulation includes a world plane and a fixed camera view. There is also a box object in the simulation that the robot can interact with.

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

#check installation path of mujoco
pip3 show mujoco | grep Location

# Build and install the plugins
cd plugin
mkdir build
cd build
cmake .. -DMUJOCO_ROOT_DIR=<filepath-from-pip-show-above>/mujoco
make install

# Install the simulation package locally
cd ../..
pip install .

# Generate the xml file for the robot using CLI
generate-baloo-xml

#run simulation
cd ../examples
python3 baloo_sim_loop.py

```

## Usage
Once everything is built and installed, you can run a simulation as shown in the examples directory. 

## Assumptions

* The inertia of the links (since this is tough to measure accurately) is assumed to be a [solid cylinder](https://en.wikipedia.org/wiki/List_of_moments_of_inertia#:~:text=%5D-,Solid%20cylinder%20of%20radius%20r%2C%20height%20h%20and%20mass%20m,-%F0%9D%90%BC)
* The mass of the joints is divided evenly (i.e. lumped evenly) between the disks that compose the joint. Each disk is assumed to be a solid cylinder as well. This also assumes that the distribution of mass is roughly uniform along the length of the joint.
* The joint_angle_estimator plugin assumes constant curvature of the joint. This is not true in the real world and is also not assumed in the simulation. The plugin is a simply estimates a constant curvature angle based on the first and last disks of the joint.
* Mostly for RL training, I used custom contact filtering using the contype and conaffinity parameters in Mujoco. The parameters are set so that the only thing that can trigger a tactile sensor response is the manipuland, not the robot itself. This is not realistic, but it is useful for training RL agents to avoid learning to punch itself.




