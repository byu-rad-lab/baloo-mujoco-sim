# Continuum Joint Angle Estimator Plugin for MuJoCo

Curtis Johnson April 2024

This C++ code defines a plugin for the MuJoCo physics engine, specifically a `JointAngleEstimator` that calculates joint angles and velocities for a given sensor instance. This mimics what we are doing right now with the VIVEs in hardware. 

## Overview

The plugin is defined within the `mujoco::plugin::sensor` namespace and includes several key methods:

- `Create`: Creates a `JointAngleEstimator` instance if all config attributes are defined and within their allowed bounds.
- `Reset`: Resets the `JointAngleEstimator` instance.
- `Compute`: Computes the joint angles and velocities.
- `Visualize`: Visualizes the joint angles and velocities.
- `RegisterPlugin`: Registers the plugin with the MuJoCo physics engine.

## Compute Method

The `Compute` method is where the joint angles (u, v) and velocities (udot, vdot) are calculated. The method first identifies the sensor id to which the plugin is attached and then calculates the joint angles and velocities.

### Joint Angles Calculation

The joint angles are calculated using the orientation of the first and last disk in the global frame. The relative orientation between the first and last disk is obtained by multiplying the conjugate of the base quaternion with the tip quaternion.

The joint angles (u, v) are then calculated using the following equations derived from Allen's paper:

```cpp
mjtNum phi = mju_acos(R_base2Tip[8]);

mjtNum u = 0;
mjtNum v = 0;
if (phi < 1e-6)
{
    u = R_base2Tip[7];
    v = R_base2Tip[2];
}
else
{
    u = R_base2Tip[7] * phi / mju_sin(phi);
    v = R_base2Tip[2] * phi / mju_sin(phi);
}
```

Here, R_base2Tip is the rotation matrix obtained from the quaternion r_base2Tip. The joint angles (u, v) are then set in the sensordata.

<!-- shwo image -->
![Joint Angle Calculation](eq5.png)

### Joint Velocities Calculation
The joint velocities are calculated by getting the sensor id for the frame angular velocity sensor on the last disk of the joint. The angular velocity of the first and last disk in the joint frame is then obtained and set in the sensordata.

## Plugin Structure
The plugin structure is defined in the RegisterPlugin method. The plugin is parameterized by the nchannel attribute and is stateless. The sensor dimension is set to 4 (u, v, udot, vdot). The plugin can only run after forces have been computed (plugin.needstage = mjSTAGE_ACC;).

The plugin includes initialization, destruction, reset, compute, and visualization callbacks. The plugin is then registered with the MuJoCo physics engine using the mjp_registerPlugin function.


## Installation
Clone this repository and build using CMAKE with the following commands:
    
  ```bash
  git clone <THIS_REPO>
  cd joint_angle_estimator
  mkdir build
  cd build
  cmake -DMUJOCO_ROOT_DIR=<PATH_TO_MUJOCO_INSTALLATION> ..
  make
  ```

<PATH_TO_MUJOCO_INSTALLATION> for me is /home/curtis/.local/lib/python3.8/site-packages/mujoco since I installed mujoco using pip. If you downloaded the mujoco code elsewhere, you need to change the path accordingly.

Then you need to install the plugin where mujoco can find it. It looks in the plugin folder of the mujoco installation. You can do this by running the following command:

  ```bash
  make install
  ```

!! IMPORTANT NOTE: Plugins must be built with the same version of mujoco that will be simulating it.