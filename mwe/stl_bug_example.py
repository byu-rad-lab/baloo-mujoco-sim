from dm_control import mjcf
import numpy as np

import mujoco
import mujoco.viewer as viewer
from regex import F


class MWE:
    def __init__(self, name) -> None:

        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self.addAssets()

        # create world plane
        self.mjcf_model.worldbody.add(
            "geom",
            condim=1,
            material="matplane",
            name="world",
            size=[0, 0, 1],
            type="plane",
        )
        self.mjcf_model.worldbody.add(
            "light",
            diffuse=[0.6, 0.6, 0.6],
            dir=[0, 0, -1],
            directional="true",
            pos=[0, 0, 4],
            specular=[0.2, 0.2, 0.2],
        )

        # add fixed camera view
        self.mjcf_model.worldbody.add(
            "camera",
            name="fixedcam",
            pos=[-1.357, 2.722, 2.447],
            xyaxes=[-0.882, -0.472, 0.000, 0.238, -0.446, 0.863],
        )

        mass = 5
        width = 0.5 / 2
        depth = 0.5 / 2
        height = 1.5 / 2
        box = self.mjcf_model.worldbody.add("body",
                                            name="box",
                                            pos=[0, 0.5, height / 2],
                                            euler=[0, 0, 0])

        box.add(
            "geom",
            name="box",
            pos=[0, 0, 0],
            type="mesh",
            mesh='test',
        )

        box.add(
            "inertial",
            pos=[0, 0, 0],
            diaginertia=[
                mass * (width**2 + depth**2) / 12,
                mass * (depth**2 + height**2) / 12,
                mass * (width**2 + height**2) / 12,
            ],
            mass=mass,
        )

        box.add("freejoint")

    def addAssets(self):
        # add children elements
        self.mjcf_model.asset.add(
            "texture",
            type="2d",
            name="texplane",
            builtin="checker",
            mark="cross",
            width=512,
            height=512,
        )

        self.mjcf_model.asset.add(
            "material",
            name="matplane",
            texture="texplane",
            texuniform="true",
        )

        # add assets for all the meshes in the meshes directory
        self.mjcf_model.asset.add(
            "mesh",
            name="test",
            file="./meshes/meshed_cube.stl",
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    torso = MWE("test")

    xml = torso.mjcf_model.to_xml_string()

    print(xml)

    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    f = open("mwe.xml", "w")
    f.write(xml)
    f.close()

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)

    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            mujoco.mj_step(model, data)

            viewer.sync()
