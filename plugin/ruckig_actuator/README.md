# Ruckig Actuator Plugin for MuJoCo

This plugin implements a Ruckig-based actuator controller for MuJoCo. It uses the Ruckig library to generate smooth trajectories for actuators based on specified constraints.

## Configuration Variables

The following table describes the configuration variables that can be set for the Ruckig actuator plugin:

| Variable           | Description                               |
| ------------------ | ----------------------------------------- |
| `max_velocity`     | The maximum velocity of the actuator.     |
| `max_acceleration` | The maximum acceleration of the actuator. |
| `max_jerk`         | The maximum jerk of the actuator.         |
| `kp`               | Proportional gain for the PID controller. |
| `kv`               | Velocity gain for the PID controller.     |
| `ka`               | Acceleration gain for the PID controller. |

## Controller Description

The Ruckig actuator plugin uses the Ruckig library to generate smooth trajectories for actuators. The controller operates as follows:

1. **Trajectory Planning**: The Ruckig library is used to generate a trajectory based on the specified constraints (`max_velocity`, `max_acceleration`, `max_jerk`) and the target position, velocity, and acceleration.
2. **PID Control**: The desired position, velocity, and acceleration from the trajectory are used to compute the control force using a PID controller with feedforward terms.

## Control Description

The control force ($F$) is computed using the following equation:

$$
F = k_p \cdot (x_d - x) + k_v \cdot (\dot{x}_d - \dot{x}) + k_a \cdot (g + \ddot{x}_d) + F_{feedforward}
$$

where:

- $x_d$ is the desired position from the trajectory.
- $x$ is the current position.
- $\dot{x}_d$ is the desired velocity from the trajectory.
- $\dot{x}$ is the current velocity.
- $\ddot{x}_d$ is the desired acceleration from the trajectory.
- $g$ is the gravitational acceleration.
- $k_p$ is the proportional gain.
- $k_v$ is the velocity gain.
- $k_a$ is the acceleration gain.
- $F_{feedforward}$ is the feedforward force computed from the gravity, inertial, coriolis, damping, and friction forces. Essentially this is a perfect feedforward term to cancel out all dynamics on the elevator (since the stepper motor is 'perfectly' stiff and throws an motor fault otherwise).


## Usage

To use the Ruckig actuator plugin, include the following configuration in your MuJoCo model file:

```xml
<!-- Example configuration for the Ruckig actuator plugin -->
<plugin type="RuckigActuator">
  <param name="max_velocity" value="1.0"/>
  <param name="max_acceleration" value="2.0"/>
  <param name="max_jerk" value="3.0"/>
  <param name="kp" value="100.0"/>
  <param name="kv" value="10.0"/>
  <param name="ka" value="1.0"/>
</plugin>
```

## Installation

To install the Ruckig actuator plugin, run the following commands from the root directory of the project:

```bash
./install.sh
```

This will download and build the Ruckig library, and then build and install the Ruckig actuator plugin.

## License

This plugin is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgements

This plugin uses the Ruckig library for trajectory generation. See [Ruckig GitHub](https://github.com/pantor/ruckig) for more information about Ruckig.