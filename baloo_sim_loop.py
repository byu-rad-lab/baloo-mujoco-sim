import time
import mujoco
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import mujoco.viewer
from baloo_mj_api import get_joint_pressures

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
        # This commented line would set random pressures in all twelve
        # chambers at every time step, currently limited to about 400 kPa,
        # but we normally operate closer to 300 kPa max.

        # set_elevator_cmd(model, data, height_cmd)
        # set_joint_pressure_commands(model, data, "right", 2, [0.0, 0.0, 400.0, 0.0])


        # update tactile image
        # image = get_tactile_image(model, data, "chest", None)
        # print(get_elevator_vel(model, data))
        # print_tactile_image(image)

        # integrate continous dynamics
        mujoco.mj_step(model, data)
        print(get_joint_pressures(model, data, 'left', 0))
        # save sensor readings, 6 angles, 6 velocties
        # left_jangles_hist.append(get_joint_angles(model, data, "left"))
        # right_jangles_hist.append(get_joint_angles(model, data, "right"))

        # left_jvel_hist.append(get_joint_vel(model, data, "left"))
        # right_jvel_hist.append(get_joint_vel(model, data, "right"))

        # height_cmd_hist.append(height_cmd)
        # height_hist.append(get_elevator_height(model, data))
        # time_hist.append(data.time)

        viewer.sync()

    viewer.close()


# # now graph/plot the data using matplotlib
# left_jangles_hist = np.array(left_jangles_hist)
# plt.figure("Left Joint angles")
# for i in range(3):
#     plt.plot(time_hist, left_jangles_hist[:, 2 * i], label=f"u{i}")
#     plt.plot(time_hist, left_jangles_hist[:, 2 * i + 1], label=f"v{i}")
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (rad)")
# plt.legend()

# right_jangles_hist = np.array(right_jangles_hist)
# plt.figure("Right Joint angles")
# for i in range(3):
#     plt.plot(time_hist, right_jangles_hist[:, 2 * i], label=f"u{i}")
#     plt.plot(time_hist, right_jangles_hist[:, 2 * i + 1], label=f"v{i}")
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (rad)")
# plt.legend()


# left_jvel_hist = np.array(left_jvel_hist)
# plt.figure("Left Joint velocity")
# for i in range(3):
#     plt.plot(time_hist, left_jvel_hist[:, 2 * i], label=f"ud{i}")
#     plt.plot(time_hist, left_jvel_hist[:, 2 * i + 1], label=f"vd{i}")
# plt.xlabel("Time (s)")
# plt.ylabel("Vel (rad/s)")
# plt.legend()

# right_jvel_hist = np.array(right_jvel_hist)
# plt.figure("Right Joint velocity")
# for i in range(3):
#     plt.plot(time_hist, right_jvel_hist[:, 2 * i], label=f"ud{i}")
#     plt.plot(time_hist, right_jvel_hist[:, 2 * i + 1], label=f"vd{i}")
# plt.xlabel("Time (s)")
# plt.ylabel("Vel (rad/s)")
# plt.legend()

# height_cmd_hist = np.array(height_cmd_hist)
# height_hist = np.array(height_hist)
# plt.figure("Height")
# plt.plot(time_hist, height_hist, label="height")
# plt.plot(time_hist, height_cmd_hist, label="height_cmd")
# plt.xlabel("Time (s)")
# plt.ylabel("Height (m)")
# plt.legend()


# pressure_hist = np.array(pressure_hist)
# commanded_pressure_hist = np.array(commanded_pressure_hist)
# plt.figure("Pressures")
# for i in range(12):
#     plt.plot(time_hist, pressure_hist[:, i], label=f"p{i}")
#     plt.plot(time_hist, commanded_pressure_hist[:, i], "--", label=f"p_cmd{i}")
# plt.xlabel("Time (s)")
# plt.ylabel("Pressure (Pa)")
# plt.legend()
# plt.show()
