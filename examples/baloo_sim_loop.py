import time
import mujoco
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import mujoco.viewer
from utils.baloo_mj_api import get_joint_pressures

# path to robot description file
XML_PATH = "/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/baloo.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)



# set printing precision to 2 decimals
np.set_printoptions(precision=2)


# nu (number of controls/actuators) can be different from na (number of activations).
# In my case, it is. So actadr is the index of the actuator's activation in data.act.
# id is the index of the actuator in the normal index in the data.ctrl vector


# # data for plotting post-simulation
height_cmd_hist = []
height_hist = []
left_jangles_hist = []
right_jangles_hist = []
left_jvel_hist = []
right_jvel_hist = []
time_hist = []


# time length of simulation
# duration = 5.0  # (seconds)

# defining start time and desired time step for visualization to render
start = data.time
height_cmd = 0.0
flag = True
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running:
        
        step_start = time.time()
        # integrate continous dynamics
        mujoco.mj_step(model, data)

        viewer.sync()

    viewer.close()



 # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)

    from utils.mjData_plotter import MjDataPlotter

    plotter = MjDataPlotter(10, model.opt.timestep)

    import time

    import mujoco.viewer

    #! this python loop can run at about .003 s/step period, so if the model has a smaller time step, you won't get real time visualization.
    #! I think this is a python/viz limitation because when I run the same model at .001s, its still 5x real time.

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

        #example of how to set the state to something specific
        set_joint_angles(model, data, 'left', 0, np.array([-0.5, -0.5]))
        set_joint_angles(model, data, 'right', 0, np.array([-0.5, -0.5]))
        set_joint_velocities(model, data, 'left', 0, np.array([-1, -1]))
        set_joint_velocities(model, data, 'right', 0, np.array([1, 1]))

        #needed to actually render the updated state
        viewer.sync()

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        print(model.body('base'))
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

            # # Rudimentary time keeping, will drift relative to wall clock.
            # # print(time.time() - step_start)
            # time_until_next_step = model.opt.timestep - (time.time() -
            #                                              step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)
