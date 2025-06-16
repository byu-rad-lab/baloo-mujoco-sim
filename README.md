# Baloo Simulation

This repository contains the code for a simulation of a robot named Baloo. 

<!-- add  scresnshot.png-->
![Sim](./screenshot.png)

## Overview

The main class in the code is `Baloo`, which represents the robot. The `Baloo` class has methods for setting up the simulation, including setting compiler options, visual settings, and contact settings. It also has methods for creating the robot's body parts and actuators. (see [```generate_baloo_xml.py```](./src/baloo_mujoco_sim/utils/generate_baloo_xml.py) for more details on how the robot is created).

The robot is composed of a number of disks, which can be specified when creating a `Baloo` instance. The robot also has a number of joints, and the height of these joints can be adjusted.

The simulation includes a world plane and a fixed camera view. There is also a box object in the simulation that the robot can interact with.

## Installation
1. Install [uv](https://docs.astral.sh/uv/) which I use a dependency manager for this project.
2. Clone the repository to your local machine.
3. Run included install script (```./install.sh```) ***in the root directory of the repository.*** This will do a few things:
   - Install the package using `uv` (this will also install the dependencies into venv)
   - Build the C++ plugins for mujoco (one for joint angle estimation and another for the elevator ruckig motion planning). The compiled files are then placed in the mujoco plugin directory. 
   - Generate the mujoco xml files for the simulation
   - Download a precompiled version of mujoco into a ```precompiled``` directory. This just comes with some useful MuJoCo binaries, like testspeed, which is useful for testing the simulation speed.
  
**NOTE: This installation has only been tested with Ubuntu 20.04 and Python 3.8. Working on updating deps to include more recent versions.** 

## Getting Started
There are two command line scripts that are installed with the package:

1. ```run-baloo-sim```: This just pulls up a passive viewer and runs the simulation. It doesn't do anything fancy, but it does allow you to see the robot in action. You can use the arrow keys to move the camera around and the mouse to zoom in and out. This CLI runs baloo_open_loop.py [here](./src/baloo_mujoco_sim/controllers/baloo_open_loop.py).
2. ```generate-baloo-xml```: Runs the xml generation script [here](./src/baloo_mujoco_sim//utils/generate_baloo_xml.py). The xml file it generates is placed in the [assets](./src/baloo_mujoco_sim/assets/) directory. This is where all of the modeling magic happens. This script reads in some parameters from [here](./src/baloo_mujoco_sim/assets/params.yaml) and included meshes from the [meshes](./src/baloo_mujoco_sim/assets/meshes/) directory.

## Python Package API

The package exposes a few different things:

1) An ```XML_PATH``` variable that points to the xml files to load a mujoco simulation.
2) A ```baloo_mj_api``` utility module that has a few helper functions for interacting with the mujoco simulation.

Example usage:

``` python
import mujoco
import baloo_mujoco_sim as baloo_mj
from baloo_mujoco_sim.utils.baloo_mj_api import set_joint_pressure_commands

# Load the simulation
 model = mujoco.MjModel.from_xml_path(baloo_mj.XML_PATH)
 data = mujoco.MjData(model)


 # later in simulation loop...
 set_joint_pressure_commands(model, data, 'left', 0, [0,0,0,0])
 ```

See [```utils/baloo_mj_api.py```](./src/baloo_mujoco_sim/utils/baloo_mj_api.py) for more details on the functions that are available. In general, the functions are for setting joint commands, getting joint angles, and getting tactile sensor readings, etc.

## Simulation Assumptions

* The inertia of the links (since this is tough to measure accurately) is assumed to be a [solid cylinder](https://en.wikipedia.org/wiki/List_of_moments_of_inertia#:~:text=%5D-,Solid%20cylinder%20of%20radius%20r%2C%20height%20h%20and%20mass%20m,-%F0%9D%90%BC)
* The mass of the joints is divided evenly (i.e. lumped evenly) between the disks that compose the joint. Each disk is assumed to be a solid cylinder as well. This also assumes that the distribution of mass is roughly uniform along the length of the joint.
* The joint_angle_estimator plugin assumes constant curvature of the joint. This is not true in the real world and is also not assumed in the simulation. The plugin is a simply estimates a constant curvature angle based on the first and last disks of the joint.
* Mostly for RL training, I used custom contact filtering using the contype and conaffinity parameters in Mujoco. The parameters are set so that the only thing that can trigger a tactile sensor response is the manipuland, not the robot itself. This is not realistic, but it is useful for training RL agents to avoid learning to punch itself.

## Future Work

### Joint Limits
Right now, the joint limits are implemented by setting the maximum bend angle in the .yaml file. This is then divided evenly between the universal joints. 

A secondary effect is that there are contacts allowed between disks[^1], which also limits the range of motion depending on how thick the disks are. Right now, the length of the joints is divided evenly between the numbers of disks + gaps. 

[^1]: Technically, its between every other disk, since by default, [MuJoCo disables contacts between geom pairs that have a parent-child body relationship](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag:~:text=down%20the%20solver.-,filterparent,-%3A%20%5Bdisable%2C%20enable). So in essence, we have contacts between parent-grandchild disks right now, which happens to land each joint at about the right limit. This could be disabled, but then the thickness of the disks needs to be adjusted to provide the correct joint limit. Disabling the parent-child filter isn't a great idea though, as it will break a bunch of other things. Alternatively, the contact between disks could be disabled completely using the ```contype``` and ```conaffinity``` parameters to rely solely on the joint limits specified in another way.

We could instead specify the maximum length for each of the tendons, which might work better. Either way, it's unclear how the actual joint limit changes with respect to the number of disks used.


### Expanding OS Compatibility

The installation script has only been tested on Ubuntu 20.04. It would be good to test on newer version of Ubuntu to ensure compatibility. 

It would also be good to add support for other operating systems, such as MacOS and Windows, though that might take more work.



