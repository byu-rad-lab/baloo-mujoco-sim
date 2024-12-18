import time
from matplotlib import pyplot as plt
import mujoco
import numpy as np
import mujoco.viewer
import baloo_mujoco_sim as baloo_mj
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_tactile_image,
    set_mocap_pose,
    set_mocap_size,
    get_disk_position,
    apply_wrench_to_body,
    clear_wrenches,
    get_contact_forces_on_body,
    detect_box_on_ground,
    set_box_position,
    set_box_size,
)


def main():
    # path to robot description file
    with open(baloo_mj.XML_PATH, 'r') as file:
        xml_string = file.read()

    mjspec = mujoco.MjSpec()
    mjspec.from_string(xml_string)

    xsize = 0.25
    ysize = 0.25
    zsize = 0.25

    # set_box_size(mjspec, xsize, ysize, zsize)

    model = mjspec.compile()
    data = mujoco.MjData(model)

    # set_box_position(model, data, 0, 1, zsize / 2)

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

        # # print(viewer.user_scn.geoms)
        # viewer.user_scn.ngeom += 1 #need to tell scene that there is a geom to render.

        # #there is a preallocated list of geoms set by maxgeom. The function below basically initializes a new geom
        # # at a specific place in this list.

        # custom_geom = viewer.user_scn.geoms[0]  #get random empty geom, and then set it to stuff.

        # mujoco.mjv_initGeom(
        #     custom_geom,
        #     type=mujoco.mjtGeom.mjGEOM_ARROW,
        #     size=[.01, .1, 1], #size in x,y,z basically, then rotation is done by mat. Length set by magnitude of force.
        #     pos=np.array([0, 2, 0]), #needs to change to match com of box.
        #     # mat=np.array([[1,0,0],[0,0,1],[0,1,0]]).flatten(),
        #     mat = np.eye(3).flatten(),
        #     rgba=np.array([0, 1, 0, 1]),
        # )
        # print(custom_geom)

        # # #sets connector like properties for the geom. don't need this if I just set everything manually.
        # # mujoco.mjv_connector(
        # #     custom_geom,
        # #     mujoco.mjtGeom.mjGEOM_ARROW,
        # #     0.1,
        # #     np.array([0, 0, 0]),
        # #     np.array([0, 1, 0]),

        # # )
        # print(custom_geom)

        #!can I change custom_geom later? since this just goes into the list of user_scn.geoms. What if I don't know which geom is at which index? Or I just have to know...
        #user scene is entirely under my control. It's separate from the main mjvScene that the viewer uses.
        # print(viewer.user_scn.geoms[0])

        #needed to actually render the updated state
        viewer.sync()

        # chest = get_tactile_image(model, data, 'chest', None)
        # plot = plt.imshow(chest, vmin=0, vmax=10, cmap='hot')
        # plt.pause(0.00001)

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # forces = get_contact_forces_on_body(model, data, 'right_link1')
            # print(f"Forces on right_link1: {forces}")

            # chest = get_tactile_image(model, data, 'chest', None)
            # plot.set_data(chest)
            # plt.pause(.0001)
            # print(viewer.user_scn.geoms[-1])

            # Example modification of a viewer option: toggle contact points every two seconds.
            # with viewer.lock():
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
            # print(viewer.perturb)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # #to render, I need to add a perturbation object.
            # apply_wrench_to_body(model, data, 'box', np.array([.1, 0, 0]), np.array([0, 0, 0]))

            # if data.time > 2:
            #     clear_wrenches(model, data)
            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == '__main__':
    main()
