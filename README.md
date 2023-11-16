# Baloo Simulation

This repository contains the code for a simulation of a robot named Baloo. The simulation is written in Python and uses the `dm_control` and `mujoco` libraries.

<!-- add  scresnshot.png-->
![Sim](screenshot.png)

## Overview

The main class in the code is `Baloo`, which represents the robot. The `Baloo` class has methods for setting up the simulation, including setting compiler options, visual settings, and contact settings. It also has methods for creating the robot's body parts and actuators.

The robot is composed of a number of disks, which can be specified when creating a `Baloo` instance. The robot also has a number of joints, and the height of these joints can be adjusted.

The simulation includes a world plane and a fixed camera view. There is also a box object in the simulation that the robot can interact with.

## Installation

To run the simulation, you will need Python and the `dm_control` and `mujoco` libraries. You can install these libraries using pip:

``` bash
pip install dm_control mujoco==3.0.1
```