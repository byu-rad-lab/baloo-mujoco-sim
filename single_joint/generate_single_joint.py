import os
import re
from attr import dataclass
from dm_control import mjcf
import numpy as np

import mujoco
import mujoco.viewer as viewer
from regex import F
from warnings import warn

import yaml
# from dm_control.mjcf.export_with_assets import export_with_assets
# from SmallJoint_mj_api import set_joint_angles, get_joint_angles


class SmallJoint:
    def __init__(self, name, num_disks, arm_angles=[]) -> None:

        warn(
            "Remember to build all plugins with current version of mujoco before running this script."
        )
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self._setupModel(num_disks)
        last_disk = self._buildSmallJoint(self.mjcf_model.worldbody, 0)

        #NOTE: doesn't look like you can put one actuator on multiple tendons. So for the 8 bellows, I'll need to create 4 big actuators, but only display the 8 for show.

        self.loadPlugins()
        self.addViveTrackers()

    def _setupModel(self, num_disks):
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self._setCompiler()
        self._setOptions()
        self._setSimSize()
        self._setVisual()
        self._addAssets()
        self._loadParams(num_disks)
        self._setCustomData()
        self._setDefaults()

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

    def _buildSmallJoint(self, parent_body, joint_num, side="left"):
        disk_mass = self.joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically).
        Ixy = (disk_mass *
               (3 * self.joint_radius**2 + self.disk_height**2)) / 12
        Iz = (disk_mass * self.joint_radius**2) / 2

        #create four spatial tendons representing the bellows
        bellows0 = self.mjcf_model.tendon.add("spatial",
                                              name=f"bellows0",
                                              dclass="tendon")
        bellows1 = self.mjcf_model.tendon.add("spatial",
                                              name=f"bellows1",
                                              dclass="tendon")
        bellows2 = self.mjcf_model.tendon.add("spatial",
                                              name=f"bellows2",
                                              dclass="tendon")
        bellows3 = self.mjcf_model.tendon.add("spatial",
                                              name=f"bellows3",
                                              dclass="tendon")

        # create first body, whose frame is offset
        first_disk = parent_body.add(
            "body",
            name=f"{side}_{joint_num}_B{0}",
            childclass="small_joint",
            pos=[0, 0, self.disk_half_height],
            euler=[0, 0, 0],
        )

        first_disk.add(
            "geom",
            name=f"{side}_{joint_num}_G0",
            dclass="small_joint",
        )
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        first_disk.add(
            "site",
            name=f"{side}_{joint_num}_xsite{0}",
            pos=[self.bellows_radius, 0, 0],
            dclass="bellows_site",
        )
        #positive x site corresponds to bellows 3
        bellows3.add("site", site=f"{side}_{joint_num}_xsite{0}")

        first_disk.add(
            "site",
            name=f"{side}_{joint_num}_-xsite{0}",
            pos=[-self.bellows_radius, 0, 0],
            dclass="bellows_site",
        )
        #negative x site corresponds to bellows 2
        bellows2.add("site", site=f"{side}_{joint_num}_-xsite{0}")

        first_disk.add(
            "site",
            name=f"{side}_{joint_num}_ysite{0}",
            pos=[0, self.bellows_radius, 0],
            dclass="bellows_site",
        )
        #positive y site corresponds to bellows 0
        bellows0.add("site", site=f"{side}_{joint_num}_ysite{0}")

        first_disk.add(
            "site",
            name=f"{side}_{joint_num}_-ysite{0}",
            pos=[0, -self.bellows_radius, 0],
            dclass="bellows_site",
        )
        #negative y site corresponds to bellows 1
        bellows1.add("site", site=f"{side}_{joint_num}_-ysite{0}")

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        # f"{side}_{joint_num}_B{i}" is format of disk bodies generally
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_{joint_num}_B{i}",
                pos=[0, 0, (2 * self.disk_height)],
            )
            body.add(
                "geom",
                name=f"{side}_{joint_num}_G{i}",
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])

            body.add("joint",
                     name=f"{side}_{joint_num}_Jx_{i-1}",
                     axis=self.X,
                     pos=[0, 0, -self.disk_height])

            body.add("joint",
                     name=f"{side}_{joint_num}_Jy_{i-1}",
                     axis=self.Y,
                     pos=[0, 0, -self.disk_height])
            prev_body = body

            # body.add(
            #     "site",
            #     name=f"rotation_center{i}",
            #     pos=[0, 0, -self.disk_height],
            #     rgba=[1, 0, 0, 1],
            # )

            body.add(
                "site",
                name=f"{side}_{joint_num}_xsite{i}",
                pos=[self.bellows_radius, 0, 0],
                dclass="bellows_site",
            )
            bellows3.add("site", site=f"{side}_{joint_num}_xsite{i}")

            body.add(
                "site",
                name=f"{side}_{joint_num}_-xsite{i}",
                pos=[-self.bellows_radius, 0, 0],
                dclass="bellows_site",
            )
            bellows2.add("site", site=f"{side}_{joint_num}_-xsite{i}")

            body.add(
                "site",
                name=f"{side}_{joint_num}_ysite{i}",
                pos=[0, self.bellows_radius, 0],
                dclass="bellows_site",
            )
            bellows0.add("site", site=f"{side}_{joint_num}_ysite{i}")

            body.add(
                "site",
                name=f"{side}_{joint_num}_-ysite{i}",
                pos=[0, -self.bellows_radius, 0],
                dclass="bellows_site",
            )
            bellows1.add("site", site=f"{side}_{joint_num}_-ysite{i}")

        self._addActuators()
        return body

    def _addActuators(self):
        #add actuator to side_tendon
        self.mjcf_model.actuator.add("cylinder",
                                     name="p0",
                                     tendon="bellows0",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        #add actuator to side_tendon
        self.mjcf_model.actuator.add("cylinder",
                                     name="p1",
                                     tendon="bellows1",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        self.mjcf_model.actuator.add("cylinder",
                                     name="p2",
                                     tendon="bellows2",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        self.mjcf_model.actuator.add("cylinder",
                                     name="p3",
                                     tendon="bellows3",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

    def _loadParams(self, num_disks):
        # some joint measurements common (hopefully) among all joints

        #load params.yaml to get physical parameters
        with open(os.path.join(os.path.dirname(__file__), "params.yaml")) as f:
            params = yaml.safe_load(f)

        self.joint_height = params['small_joint']['height']
        self.joint_mass = params['small_joint']['mass']
        self.joint_radius = params['small_joint']['radius']
        self.bellows_radius = params['small_joint']['bellows_radius']
        self.joint_limit = params['small_joint']['bend_limit']
        self.joint_stiffness = params['small_joint']['lumped_stiffness']
        self.joint_damping = params['small_joint']['lumped_damping']

        self.num_joints = num_disks - 1
        self.num_disks = num_disks
        num_spaces = self.num_disks - 1
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2

    def loadPlugins(self):
        plugin = self.mjcf_model.extension.add(
            "plugin",
            plugin="mujoco.sensor.joint_angle_estimator",
        )

    def addViveTrackers(self):
        # add framequat to base and tip ref to global frame
        self.mjcf_model.sensor.add(
            "framequat",
            name="left_0_B0_framequat",
            objtype="body",
            objname="left_0_B0",
        )

        self.mjcf_model.sensor.add(
            "framequat",
            name=f"left_0_B{self.num_disks-1}_framequat",
            objtype="body",
            objname=f"left_0_B{self.num_disks-1}",
        )

        #add frameangvel to tip, ref to base frame
        self.mjcf_model.sensor.add(
            "frameangvel",
            name=f"left_0_B{self.num_disks-1}_frameangvel",
            objtype="body",
            objname=f"left_0_B{self.num_disks-1}",
            reftype="body",
            refname="left_0_B0",
        )

        self.mjcf_model.sensor.add(
            "plugin",
            plugin="mujoco.sensor.joint_angle_estimator",
            name='left_0')

    def _setCustomData(self):
        self.mjcf_model.custom.add(
            "numeric",
            name="num_disks",
            size=1,
            data=[self.num_disks],
        )

    def _setCompiler(self):
        self.mjcf_model.compiler.angle = "degree"

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

    def _setDefaults(self):
        # damping uniformly distributed over each disk
        # dampers in series, so d_lumped = d_disk/num_joints
        # stiffness is lumped over all bellows
        four_stiffness = self.joint_stiffness * self.num_joints
        four_damping = self.joint_damping * self.num_joints

        small_disk_class = self.mjcf_model.default.add("default",
                                                       dclass="small_joint")
        small_disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[self.joint_radius, self.disk_half_height * 0.5],
        )
        small_disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            damping=four_damping,
            pos=[0, 0, self.disk_height],
            limited="false",
            stiffness=four_stiffness,
        )

        bellows_site_class = self.mjcf_model.default.add("default",
                                                         dclass="bellows_site")
        bellows_site_class.site.set_attributes(rgba=[0, 1, 0, 1])

        tendon_class = self.mjcf_model.default.add("default", dclass="tendon")

        tendon_class.tendon.set_attributes(width=0.03, )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    torso = SmallJoint(
        "SmallJoint",
        5,
    )

    # print(torso.mjcf_model)
    # export_with_assets(torso.mjcf_model,'.', "SmallJoint.xml", zero_threshold=1e-8)

    xml = torso.mjcf_model.to_xml_string()

    # bandaid for weird bug to replace strings inserted after file names:
    # remove random letters and numbers in between dash and .stl from comments above
    xml = re.sub(r"-(.*?).stl", ".stl", xml)

    # prepend absolute path to all stl in xml file
    xml = re.sub(
        r"file=\"",
        'file="/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/SmallJoint/meshes/',
        xml,
    )

    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    f = open("single_joint.xml", "w")
    f.write(xml)
    f.close()

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)
    import time
    from plotter import JointAnglePlotter
    plotter = JointAnglePlotter()
    plotter.show()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        # with viewer.lock():q
        #     #update gui things
        #     mujoco.mjr_figure(viewport, fig, context)

        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # plotter.update(model, data)

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        plotter.close()
