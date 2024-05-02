from dm_control import mjcf
import numpy as np
import mujoco
import mujoco.viewer as viewer


class SingleDOF:
    def __init__(self, name) -> None:
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self._setupModel()
        self._buildPendMassDamper(self.mjcf_model.worldbody)

    def _setupModel(self):
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self._setCompiler()
        self._setOptions()
        self._setSimSize()
        self._setVisual()
        self._addAssets()

        # create checkered world plane
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

    def _setCompiler(self):
        self.mjcf_model.compiler.angle = "degree"
        self.mjcf_model.compiler.inertiafromgeom = "true"

    def _setOptions(self):
        self.mjcf_model.option.set_attributes(
            timestep=0.01,
            integrator='implicitfast',  #recommended by mujoco docs as best
            solver="Newton",
            jacobian="sparse",
            cone="elliptic",
        )

        self.mjcf_model.option.flag.set_attributes(gravity="enable")

    def _setSimSize(self):
        self.mjcf_model.size.set_attributes(njmax=5000,
                                            nconmax=5000,
                                            nstack=5000000)

    def _setVisual(self):
        # visual already has all possible children elements created, so just change them here.
        self.mjcf_model.visual.map.set_attributes(stiffness=100,
                                                  fogstart=10,
                                                  fogend=15,
                                                  zfar=40,
                                                  shadowscale=0.5)
        self.mjcf_model.visual.scale.set_attributes(
            forcewidth=0.1,
            contactwidth=0.3 * 0.25,
            contactheight=0.1 * 0.25,
            framelength=1.0 * 0.6,
            framewidth=0.1 * 0.6,
        )

    def _addAssets(self):
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

    def _buildPendMassDamper(self, parent_body):
        pend = parent_body.add(
            'body',
            name='pendulum',
            pos=[0, 0, 0.1],
            euler=[0, 90, 0],
        )

        pend.add(
            "geom",
            type="capsule",
            size=[0.05, 0.5],
            fromto=[0, 0, 0, 0, 0, 1],
        )

        m = 1
        l = 1
        I = m * l**2 / 12
        pend.add(
            "inertial",
            mass=m,
            pos=[0, 0, l / 2],
        )

        pend.add(
            "joint",
            name="joint",
            pos=[0, 0, 0],
            axis=[1, 0, 0],
            springdamper=[0.5, 0.2],
            # frictionloss=10,
        )

        self.mjcf_model.actuator.add(
            "position",
            name='velocity_servo',
            joint="joint",
            kp=500,
            kv=50,
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    test = SingleDOF("test")

    # print(torso.mjcf_model)
    # export_with_assets(torso.mjcf_model,'.', "baloo.xml", zero_threshold=1e-8)

    xml = test.mjcf_model.to_xml_string()

    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    f = open("1dof.xml", "w")
    f.write(xml)
    f.close()

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)

    import time

    import mujoco.viewer

    #! this python loop can run at about .003 s/step period, so if the model has a smaller time step, you won't get real time visualization.
    #! I think this is a python/viz limitation because when I run the same model at .001s, its still 5x real time.

    with mujoco.viewer.launch_passive(model, data) as viewer:
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

            # print(data.sensor('left_0').data)

            # if detect_box_touch(model, data):
            # print("box touched at time: ", data.time)
            # get_contact_force(model, data)

            # contact_forces = get_contact_forces_on_body(model, data, "box")
            # print(f"net force on box: {contact_forces.sum(axis=0)}")
            # print(f"contact forces on box\n: {contact_forces}")

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
