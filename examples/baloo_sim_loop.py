import time
import mujoco
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import mujoco.viewer
import baloo_mujoco_sim as baloo_mj
from baloo_mujoco_sim.utils.mjData_plotter import MjDataPlotter

# path to robot description file
model = mujoco.MjModel.from_xml_string(baloo_mj.XML_STRING)
data = mujoco.MjData(model)

plotter = MjDataPlotter(10, model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # with viewer.lock():
    # disable shadows and reflectionas to boost frame rate
    # viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    # viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    #
    # set transparent bodies, and contact points for better visualization
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # viewer.opt.label = mujoco.mjtLabel.mjLABEL_CONTACTPOINT

    # #example of how to set the state to something specific
    # set_joint_angles(model, data, 'left', 0, np.array([-0.5, -0.5]))
    # set_joint_angles(model, data, 'right', 0, np.array([-0.5, -0.5]))
    # set_joint_velocities(model, data, 'left', 0, np.array([-1, -1]))
    # set_joint_velocities(model, data, 'right', 0, np.array([1, 1]))

    #needed to actually render the updated state
    viewer.sync()

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # plotter.update(
        #     model, data)  # takes almost as long to plot as to simulate.

        # Rudimentary time keeping, will drift relative to wall clock.
        # print(time.time() - step_start)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
