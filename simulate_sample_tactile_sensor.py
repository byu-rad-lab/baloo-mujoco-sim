import mujoco
import mujoco.viewer
import time

# path to robot description file
# XML_PATH = "/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/touch_grid.xml"
XML_PATH = "/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/sample_tactile_sensor.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

"""
Need to build repository https://github.com/isri-aist/MujocoTactileSensorPlugin
then:

1) find where python installed mujoco with pip3 show mujoco
2) move .so file into plugin folder inside the mujoco folder
3) then this works. but only from python. It doesn't work from /bin folder of a mujoco download since the mujoco executbale is different.
"""

# should be used this way, but I need passive viewer to work.
# mujoco.viewer.launch(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        # disable shadows and reflectionas to boost frame rate
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        # set transparent bodies, and contact points for better visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_CONTACTPOINT

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

        # if detect_box_touch(model, data):
        # print("box touched at time: ", data.time)
        # get_contact_force(model, data)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
