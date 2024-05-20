from dm_control import mjcf
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer


class TactileGrid:
    def __init__(self, name, N_width, N_height) -> None:
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self.ORANGE = [0.8, 0.2, 0.1, 1]
        self.VENTION_BLUE = [0, 40 / 255, 80 / 255, 1]
        self.BLACK = [0 / 255, 0 / 255, 0 / 255, 1]
        self.WHITE = [255, 255, 255, 1]
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        # self.ORANGE = [15 / 256.0, 10 / 256.0, 222 / 256.0, 1]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self.setCompiler()
        self.setOptions()
        self.setSimSize()
        self.setVisual()
        self.addAssets()

        # self.setDefaults()
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
        self.addTaxels(self.mjcf_model.worldbody)
        self.addBall(self.mjcf_model.worldbody)

    def addBall(self, parent):
        ball = parent.add(
            "body", name="ball", pos=[-0.10, -0.10, 0.25], euler=[0, 0, 0]
        )

        ball.add(
            "geom",
            type="box",
            name="ball",
            pos=[0, 0, 0],
            size=[0.15 / 2, 0.15 / 2, 0.15 / 2],
            # size=[.25/2],
            rgba=self.VENTION_BLUE,
        )

        ball.add("inertial", pos=[0, 0, 0], mass=1000, diaginertia=[100, 100, 100])

        ball.add("joint", type="free")

    def addTaxels(self, base):
        # ? not sure if geoms should be added to base directly, or to other things. We shall see.
        l = 0.01
        x = 0.25
        y = 0.25
        z = 0
        dy = 0
        for i in range(30):
            dx = 0
            for j in range(30):
                site_name = f"taxel_{i}_{j}"
                location = [(x / 2 - l / 2) - dx, (y / 2 - l / 2) - dy, z + l / 2]
                # base.add(
                #     "geom",
                #     name=f"geom_{i}_{j}",
                #     type="sphere",
                #     # size=[l / 2, l / 2, l / 2],
                #     size=[l / 2],
                #     rgba=self.ORANGE,
                #     pos=location,
                # )

                base.add(
                    "site",
                    name=site_name,
                    type="sphere",
                    # size=[l / 2, l / 2, l / 2],
                    size=[l / 2],
                    pos=location,
                    rgba=self.ORANGE,
                )

                # add sensor to this site
                self.mjcf_model.sensor.add("touch", site=site_name)

                dx += l * 1.5
            dy += l * 1.5

    def createBase(self, body):
        # create linear actuator and torso from which to hang arms
        base = body.add("body", name="base", pos=[0, 0, 0], euler=[0, 0, 0])
        # add geom

        actuator_height = 0.25
        base.add(
            "geom",
            name="base",
            pos=[0, 0, actuator_height / 2],
            type="box",
            size=[1 / 2, 1 / 2, actuator_height / 2],
            rgba=self.VENTION_BLUE,
        )

        return base

    def setCompiler(self):
        self.mjcf_model.compiler.angle = "degree"

    def setOptions(self):
        self.mjcf_model.option.set_attributes(
            timestep=0.01,
            iterations=10,
            solver="Newton",
            jacobian="sparse",
            cone="elliptic",
            tolerance=1e-10,
        )

        self.mjcf_model.option.flag.set_attributes(gravity="enable")

    def setSimSize(self):
        self.mjcf_model.size.set_attributes(njmax=5000, nconmax=10000, nstack=50000)

    def setVisual(self):
        # visual already has all possible children elements created, so just change them here.
        self.mjcf_model.visual.map.set_attributes(
            stiffness=100, fogstart=10, fogend=15, zfar=40, shadowscale=0.5
        )

        self.mjcf_model.visual.scale.set_attributes(
            forcewidth=0.1 * 0.05,
            contactwidth=0.3 * 0.05,
            contactheight=0.1 * 0.05,
            framelength=1.0 * 0.3,
            framewidth=0.1 * 0.3,
        )

    def addAssets(self):
        # add children elements
        self.mjcf_model.asset.add(
            "texture",
            type="2d",
            name="texplane",
            builtin="checker",
            mark="cross",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.15, 0.2],
            markrgb=[0.8, 0.8, 0.8],
            width=512,
            height=512,
        )

        self.mjcf_model.asset.add(
            "material",
            name="matplane",
            texture="texplane",
            texuniform="true",
            reflectance=0.3,
        )


if __name__ == "__main__":
    tactile_grid = TactileGrid("tactile_sleeve", 10, 10)

    xml = tactile_grid.mjcf_model.to_xml_string()

    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    f = open("tactile_sleeve.xml", "w")
    f.write(xml)
    f.close()

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)

    # viewer block so user can interact with object in GUI.
    viewer.launch(model, data)
