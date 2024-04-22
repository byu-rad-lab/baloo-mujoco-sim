import os
import re
from attr import dataclass
from dm_control import mjcf
import numpy as np

import mujoco
import mujoco.viewer as viewer
from regex import F
from warnings import warn
# from dm_control.mjcf.export_with_assets import export_with_assets
# from baloo_mj_api import set_joint_angles, get_joint_angles


class Baloo:
    def __init__(self, name, num_disks, arm_angles=[]) -> None:

        warn(
            "Remember to build all plugins with current version of mujoco before running this script."
        )
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self.ORANGE = [0.8, 0.2, 0.1, 1]
        self.VENTION_BLUE = [0, 40 / 255, 80 / 255, 1]
        self.BLACK = [0 / 255, 0 / 255, 0 / 255, 1]
        self.WHITE = [255, 255, 255, 1]
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        self.GRAY2 = [50 / 255, 50 / 255, 50 / 255, 0.7]
        # self.ORANGE = [15 / 256.0, 10 / 256.0, 222 / 256.0, 1]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self.setCompiler()
        self.setOptions()
        self.setSimSize()
        self.setVisual()
        self.addAssets()
        # self.setContacts()

        # some joint measurements common (hopefully) among all joints
        self.joint_height = 0.2  # length between endplates (m)
        self.num_joints = 1
        self.num_disks = num_disks
        num_spaces = self.num_disks - 1
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2

        self.setCustomData()
        self.setDefaults()

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

        # # TODO: add createObject to be manipulated here.
        # # pos = (-.5, .5)m, box of .5 m side,
        # mass = 5
        # width = 0.5 / 2
        # depth = 0.5 / 2
        # height = 1.5 / 2
        # box = self.mjcf_model.worldbody.add("body",
        #                                     name="box",
        #                                     pos=[0, 0.5, height / 2],
        #                                     euler=[0, 0, 0])

        # box.add(
        #     "geom",
        #     name="box",
        #     pos=[0, 0, 0],
        #     type="box",
        #     size=[width / 2, depth / 2, height / 2],
        #     rgba=self.ORANGE,
        # )

        # box.add(
        #     "inertial",
        #     pos=[0, 0, 0],
        #     diaginertia=[
        #         mass * (width**2 + depth**2) / 12,
        #         mass * (depth**2 + height**2) / 12,
        #         mass * (width**2 + height**2) / 12,
        #     ],
        #     mass=mass,
        # )

        # box.add("freejoint")

        # linear_actuator = self.createBase(self.mjcf_model.worldbody)
        # chest = self.createChest(linear_actuator)
        # right_shoulder, left_shoulder = self.createShoulders(chest)

        # right_last_disk = self.createLargeJoint(right_shoulder, "right")
        # right_link0 = self.addLink0(right_last_disk, "right")
        # last_disk = self.createMediumJoint(right_link0, "right")
        # right_link1 = self.addLink1(last_disk, "right")
        last_disk = self.createSmallJoint(self.mjcf_model.worldbody)

        #create spatial tendon
        side_tendon = self.mjcf_model.tendon.add("spatial",
                                                 name=f"xtest",
                                                 width=0.03)
        side_tendon.add("site", site="xsite0")
        side_tendon.add("site", site="xsite1")
        side_tendon.add("site", site="xsite2")
        side_tendon.add("site", site="xsite3")
        side_tendon.add("site", site="xsite4")

        side_tendon = self.mjcf_model.tendon.add("spatial",
                                                 name=f"-xtest",
                                                 width=0.03)
        side_tendon.add("site", site="-xsite0")
        side_tendon.add("site", site="-xsite1")
        side_tendon.add("site", site="-xsite2")
        side_tendon.add("site", site="-xsite3")
        side_tendon.add("site", site="-xsite4")

        side_tendon = self.mjcf_model.tendon.add("spatial",
                                                 name=f"ytest",
                                                 width=0.03)
        side_tendon.add("site", site="ysite0")
        side_tendon.add("site", site="ysite1")
        side_tendon.add("site", site="ysite2")
        side_tendon.add("site", site="ysite3")
        side_tendon.add("site", site="ysite4")

        side_tendon = self.mjcf_model.tendon.add("spatial",
                                                 name=f"-ytest",
                                                 width=0.03)
        side_tendon.add("site", site="-ysite0")
        side_tendon.add("site", site="-ysite1")
        side_tendon.add("site", site="-ysite2")
        side_tendon.add("site", site="-ysite3")
        side_tendon.add("site", site="-ysite4")

        #add actuator to side_tendon
        self.mjcf_model.actuator.add("cylinder",
                                     name="p0",
                                     tendon="xtest",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        #add actuator to side_tendon
        self.mjcf_model.actuator.add("cylinder",
                                     name="p1",
                                     tendon="-xtest",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        self.mjcf_model.actuator.add("cylinder",
                                     name="p2",
                                     tendon="ytest",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        self.mjcf_model.actuator.add("cylinder",
                                     name="p3",
                                     tendon="-ytest",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

        #NOTE: doesn't look like you can put one actuator on multiple tendons. So for the 8 bellows, I'll need to create 4 big actuators, but only display the 8 for show.

        # side_tendon.add("site", site="site5")

        # self.createActuators("right")

        # left_last_disk = self.createLargeJoint(left_shoulder, "left")
        # left_link0 = self.addLink0(left_last_disk, "left")
        # last_disk = self.createMediumJoint(left_link0, "left")
        # left_link1 = self.addLink1(last_disk, "left")
        # last_disk = self.createSmallJoint(left_link1, "left")
        # self.createActuators("left")

        # self.createSensors()

        self.addPlugins()
        self.addViveTrackers()

    def addPlugins(self):
        plugin = self.mjcf_model.extension.add(
            "plugin",
            plugin="mujoco.sensor.joint_angle_estimator",
            # instance="joint_angle_estimator",
        )

        plugin.add("instance", name="test")

    def addViveTrackers(self):
        # add framequat to base and tip ref to global frame
        self.mjcf_model.sensor.add(
            "framequat",
            name="0_B0_framequat",
            objtype="body",
            objname="0_B0",
        )

        self.mjcf_model.sensor.add(
            "framequat",
            name=f"0_B{self.num_disks-1}_framequat",
            objtype="body",
            objname=f"0_B{self.num_disks-1}",
        )

        #add frameangvel to tip, ref to base frame
        self.mjcf_model.sensor.add(
            "frameangvel",
            name=f"0_B{self.num_disks-1}_frameangvel",
            objtype="body",
            objname=f"0_B{self.num_disks-1}",
            reftype="body",
            refname="0_B0",
        )

        self.mjcf_model.sensor.add(
            "plugin",
            plugin="mujoco.sensor.joint_angle_estimator",
            instance="test",
            name='vive_tracker')

    def setCustomData(self):
        self.mjcf_model.custom.add(
            "numeric",
            name="num_disks",
            size=1,
            data=[self.num_disks],
        )

    def setContacts(self):
        self.mjcf_model.contact.add(
            "exclude",
            name="left0",
            body1="world",
            body2="left_link0",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="left1",
            body1="world",
            body2="left_link1",
        )
        self.mjcf_model.contact.add(
            "exclude",
            name="right0",
            body1="world",
            body2="right_link0",
        )
        self.mjcf_model.contact.add(
            "exclude",
            name="right1",
            body1="world",
            body2="right_link1",
        )

    def createActuators(self, side):
        # I use fixed tendons to be able to actuate all the little disk joints together.
        time_consts = [0.2, 0.5, 0.8]  # based on size of valve mostly
        gears = [
            0.05 * 1000,
            0.05 * 1000,
            0.05 * 1000,
        ]  # distance from center of joint to center of bellows, moment arms
        diameters = [
            0.05,
            0.05,
            0.05,
        ]  # sqrt(2) doubles area for big joint

        Psrc = 40  # psi
        maxP = Psrc * 6.895  # kpa

        for j in range(self.num_joints):
            # create both x and y tendons
            xtendon = self.mjcf_model.tendon.add("fixed", name=f"x{j}")
            ytendon = self.mjcf_model.tendon.add("fixed", name=f"y{j}")

            # add actuators along tendons
            # ====== X AXIS =========
            # bellows that creates positive rotation
            self.mjcf_model.actuator.add(
                "cylinder",
                name=f"j{j}_p0",
                tendon=f"x{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[gears[j]],
                timeconst=time_consts[j],
            )
            # bellows that creates negative rotation, switched using gear
            self.mjcf_model.actuator.add(
                "cylinder",
                name=f"j{j}_p1",
                tendon=f"x{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[-gears[j]],
                timeconst=time_consts[j],
            )

            # ========= Y AXIS ===========
            self.mjcf_model.actuator.add(
                "cylinder",
                name=f"j{j}_p2",
                tendon=f"y{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[gears[j]],
                timeconst=time_consts[j],
            )
            # bellows that creates negative rotation, switched using gear
            self.mjcf_model.actuator.add(
                "cylinder",
                name=f"j{j}_p3",
                tendon=f"y{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[-gears[j]],
                timeconst=time_consts[j],
            )

            # tendon length is the sum of joint angles between each disk. (i.e. q_lumped)
            # num_disks - 1 b.c. there are num_disks - 1 spaces between disks
            for n in range(0, self.num_disks - 1):
                xtendon.add("joint", joint=f"{j}_Jx_{n}", coef=1)
                ytendon.add("joint", joint=f"{j}_Jy_{n}", coef=1)

    def createSensors(self):
        # the values for these sensors are in mjData/sensordata, which is stored as an array of nsensordata x 1
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qx0",
                                   tendon="left_x0")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qy0",
                                   tendon="left_y0")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qx1",
                                   tendon="left_x1")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qy1",
                                   tendon="left_y1")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qx2",
                                   tendon="left_x2")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="left_qy2",
                                   tendon="left_y2")

        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qx0",
                                   tendon="right_x0")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qy0",
                                   tendon="right_y0")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qx1",
                                   tendon="right_x1")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qy1",
                                   tendon="right_y1")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qx2",
                                   tendon="right_x2")
        self.mjcf_model.sensor.add("tendonpos",
                                   name="right_qy2",
                                   tendon="right_y2")
        #

        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qxd0",
                                   tendon="left_x0")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qyd0",
                                   tendon="left_y0")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qxd1",
                                   tendon="left_x1")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qyd1",
                                   tendon="left_y1")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qxd2",
                                   tendon="left_x2")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="left_qyd2",
                                   tendon="left_y2")

        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_qxd0",
                                   tendon="right_x0")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_qyd0",
                                   tendon="right_y0")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_qxd1",
                                   tendon="right_x1")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_qyd1",
                                   tendon="right_y1")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_qxd2",
                                   tendon="right_x2")
        self.mjcf_model.sensor.add("tendonvel",
                                   name="right_vd2",
                                   tendon="right_y2")

        # self.mjcf_model.sensor.add("actuatorfrc", name="act_force", actuator="p0_j0")

    def addLink0(self, body, side):
        link = body.add(
            "body",
            name=f"{side}_link0",
            pos=[0, 0, -(self.disk_half_height + 0.1)],
            euler=[0, 0, -45],
        )
        link.add("inertial",
                 pos=[0, 0, 0],
                 diaginertia=[0.108, 0.108, 0.023],
                 mass=3.881)

        # TODO: fix this inertial frame pos to account for valve block and stuff. Not sure what pos is relative to.
        link.add(
            "geom",
            name=f"{side}_link0",
            type="cylinder",
            size=[0.13, 0.1],
            rgba=self.BLACK,
        )

        link_height = 0.2

        r = 0.13
        self.add_tactile_sleeve(side, link, 0, link_height, r)

        return link

    def add_tactile_sleeve(self, side, link, linknum, link_height, r):
        # need to add site to attach sensor and geom to generate collision to the body
        # need function for cylinders of some radius, height, and spacing (since we know its a 16x64 taxel array)
        # todo: need to scale size of geoms based on geometry since taxels size is different.
        theta = 0.0
        dtheta = 360 / 64
        height = link_height / 2 - (link_height / 16) * 0.5
        dh = link_height / 16
        for i in range(64):
            h = height
            for j in range(16):
                site_name = f"{side}_link{linknum}_{i}_{j}"
                link.add(
                    "geom",
                    name=site_name + "_geom",
                    type="sphere",
                    size=[0.01 / 2],
                    pos=[
                        r * np.cos(np.radians(theta)),
                        r * np.sin(np.radians(theta)),
                        h,
                    ],
                    euler=[0, 0, theta],
                    rgba=self.GRAY2,
                )
                link.add(
                    "site",
                    name=site_name,
                    type="sphere",
                    size=[0.01 / 2],
                    pos=[
                        r * np.cos(np.radians(theta)),
                        r * np.sin(np.radians(theta)),
                        h,
                    ],
                    euler=[0, 0, theta],
                    rgba=self.GRAY2,
                )

                # add sensor to this site
                self.mjcf_model.sensor.add(
                    "touch",
                    name=f"{side}_link{linknum}_{i}_{j}_touch",
                    site=site_name)

                h -= dh
            theta += dtheta

    def addLink1(self, body, side):
        link = body.add(
            "body",
            name=f"{side}_link1",
            pos=[0, 0, -(self.disk_half_height + 0.08)],
            euler=[0, 0, -45],
        )
        link.add(
            "inertial",
            pos=[0, 0, 0],
            diaginertia=[0.05, 0.05, 0.017],
            mass=3.474,
        )

        # TODO: fix this inertial frame pos to account for valve block and stuff. Not sure what pos is relative to.
        link.add(
            "geom",
            name=f"{side}_link1",
            type="cylinder",
            size=[0.1, 0.08],
            rgba=self.BLACK,
        )

        r = 0.1
        link_height = 0.08 * 2

        self.add_tactile_sleeve(side, link, 1, link_height, r)

        return link

    def createLargeJoint(self, body, side):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        joint_radius = 0.125
        joint_mass = 2.653
        num_spaces = self.num_disks - 1
        disk_height = self.joint_height / (self.num_disks + num_spaces)
        disk_half_height = disk_height / 2
        disk_mass = joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass * (3 * joint_radius**2 + disk_height**2)) / 12
        Iz = (disk_mass * joint_radius**2) / 2

        # create first body, whose frame is offset
        joint_num = 0
        first_disk = body.add(
            "body",
            name=f"{side}_{joint_num}_B0",
            childclass="large_joint",
            pos=[0, 0, -(0.254 / 2 + disk_half_height)],
            euler=[0, 0, 45],
        )
        first_disk.add(
            "geom",
            name=f"{side}_{joint_num}_G0",
        )
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_{joint_num}_B{i}",
                pos=[0, 0, -(2 * disk_height)],
            )
            body.add(
                "geom",
                name=f"{side}_{joint_num}_G{i}",
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])
            body.add("joint", name=f"{side}_{joint_num}_Jx_{i-1}", axis=self.X)
            body.add("joint", name=f"{side}_{joint_num}_Jy_{i-1}", axis=self.Y)
            prev_body = body

        return body

    def createMediumJoint(self, body, side):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        joint_mass = 1.326
        joint_radius = 0.3
        num_spaces = self.num_disks - 1
        disk_height = self.joint_height / (self.num_disks + num_spaces)
        disk_half_height = disk_height / 2
        disk_mass = joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass * (3 * joint_radius**2 + disk_height**2)) / 12
        Iz = (disk_mass * joint_radius**2) / 2

        # create first body, whose frame is offset
        joint_num = 1
        first_disk = body.add(
            "body",
            name=f"{side}_{joint_num}_B0",
            childclass="medium_joint",
            pos=[0, 0, -(0.1 + disk_half_height)],  # from pneubotics
            euler=[0, 0, 45],
        )
        first_disk.add(
            "geom",
            name=f"{side}_{joint_num}_G0",
        )
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom, add joints
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_{joint_num}_B{i}",
                pos=[0, 0, -(2 * disk_height)],
            )
            body.add(
                "geom",
                name=f"{side}_{joint_num}_G{i}",
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])

            #creates motion dof between body and the body's parent (i.e. prev_body)
            body.add("joint", name=f"{side}_{joint_num}_Jx_{i-1}", axis=self.X)
            body.add("joint", name=f"{side}_{joint_num}_Jy_{i-1}", axis=self.Y)
            prev_body = body

        return body

    def createSmallJoint(self, body):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        joint_mass = 1.326
        joint_radius = 0.08
        num_spaces = self.num_disks - 1
        disk_height = self.joint_height / (self.num_disks + num_spaces)
        disk_half_height = disk_height / 2
        disk_mass = joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass * (3 * joint_radius**2 + disk_height**2)) / 12
        Iz = (disk_mass * joint_radius**2) / 2

        # create first body, whose frame is offset
        joint_num = 0
        first_disk = body.add(
            "body",
            name=f"{joint_num}_B0",
            childclass="small_joint",
            pos=[0, 0, (disk_half_height)],  # from pneubotics
            euler=[0, 0, 0],
        )
        first_disk.add(
            "geom",
            name=f"{joint_num}_G0",
        )
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        first_disk.add(
            "site",
            name=f"xsite{0}",
            pos=[joint_radius / 2, 0, 0],
            rgba=[0, 1, 0, 1],
        )

        first_disk.add(
            "site",
            name=f"-xsite{0}",
            pos=[-joint_radius / 2, 0, 0],
            rgba=[0, 1, 0, 1],
        )

        first_disk.add(
            "site",
            name=f"ysite{0}",
            pos=[0, joint_radius / 2, 0],
            rgba=[0, 1, 0, 1],
        )

        first_disk.add(
            "site",
            name=f"-ysite{0}",
            pos=[0, -joint_radius / 2, 0],
            rgba=[0, 1, 0, 1],
        )

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{joint_num}_B{i}",
                pos=[0, 0, (2 * disk_height)],
            )
            body.add(
                "geom",
                name=f"{joint_num}_G{i}",
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])
            body.add("joint",
                     name=f"{joint_num}_Jx_{i-1}",
                     axis=self.X,
                     pos=[0, 0, -disk_height])
            body.add("joint",
                     name=f"{joint_num}_Jy_{i-1}",
                     axis=self.Y,
                     pos=[0, 0, -disk_height])
            prev_body = body

            body.add(
                "site",
                name=f"rotation_center{i}",
                pos=[0, 0, -disk_height],
                rgba=[1, 0, 0, 1],
            )

            body.add(
                "site",
                name=f"xsite{i}",
                pos=[joint_radius / 2, 0, 0],
                rgba=[0, 1, 0, 1],
            )

            body.add(
                "site",
                name=f"-xsite{i}",
                pos=[-joint_radius / 2, 0, 0],
                rgba=[0, 1, 0, 1],
            )
            body.add(
                "site",
                name=f"ysite{i}",
                pos=[0, joint_radius / 2, 0],
                rgba=[0, 1, 0, 1],
            )

            body.add(
                "site",
                name=f"-ysite{i}",
                pos=[0, -joint_radius / 2, 0],
                rgba=[0, 1, 0, 1],
            )

        return body

    def createBase(self, body):
        # create linear actuator and torso from which to hang arms
        base = body.add("body", name="base", pos=[0, 0, 0], euler=[0, 0, 0])

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LeftBaseMesh",
        #     material="vention_blue",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="RightBaseMesh",
        #     material="vention_blue",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="BaseFrameMesh",
        #     material="vention_blue",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LinearActuatorMesh",
        #     material="vention_blue",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="PneumaticInletMesh",
        #     material="silver",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="PowerButtonMesh",
        #     material="red",
        # )

        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="ControlBoxMesh",
        #     material="matte_black",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="EstopPlugMesh",
        #     material="green",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="EthernetJackMesh",
        #     material="silver",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LCDScreenMesh",
        #     material="lcd_blue",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LeftBackWheelFootMesh",
        #     material="matte_black",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LeftFrontWheelFootMesh",
        #     material="matte_black",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="RightBackWheelFootMesh",
        #     material="matte_black",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="RightFrontWheelFootMesh",
        #     material="matte_black",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LeftBackWheelMesh",
        #     material="cream",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="LeftFrontWheelMesh",
        #     material="cream",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="RightBackWheelMesh",
        #     material="cream",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="RightFrontWheelMesh",
        #     material="cream",
        # )
        # base.add(
        #     "geom",
        #     type="mesh",
        #     mesh="StepperMesh",
        #     material="matte_black",
        # )
        return base

    def createChest(self, linear_actuator):
        chest = linear_actuator.add("body",
                                    name="chest",
                                    pos=[0, 0.1945, 1.4],
                                    euler=[0, 0, 0])

        # add geom
        # chest.add("geom", name="chest", pos=[0, 0, 0], type="mesh", mesh="chest")
        chest.add(
            "geom",
            name="chest",
            type="mesh",
            mesh="SimpleChestMesh",
            material="silver",
        )

        # add inertial properties
        chest_mass = 5
        chest_width = 0.500
        chest_height = 0.254
        chest_depth = 0.254
        chest.add(
            "inertial",
            mass=chest_mass,
            diaginertia=[
                chest_mass * (chest_width**2 + chest_depth**2) / 12,
                chest_mass * (chest_depth**2 + chest_height**2) / 12,
                chest_mass * (chest_width**2 + chest_height**2) / 12,
            ],
            pos=[0, 0, 0],
        )

        chest.add(
            "joint",
            name="linear_actuator",
            type="slide",
            axis=[0, 0, 1],
            limited=True,
            range=[-1.2, 0],
            damping=500,
        )

        # TODO: not sure how to model this ball screw joint correctly. Acording to
        # https://github.com/deepmind/mujoco/issues/175, this is correct. But I don't have any guarantee that the
        # trapezoidal vel profile is actually followed doing it this way.

        # but having enough damping to keep it slow enough causes a lot of steady state error. Don't love this.
        self.mjcf_model.actuator.add(
            "position",
            name=f"elevator",
            joint="linear_actuator",
            ctrllimited=True,
            ctrlrange=[-1, 0],
            kp=1000,
            # forcerange=[-300, 800],
            # forcelimited=True,
        )

        # add tactile sensors to front of chest 30 rows, 16 columns for one side (32 columns for both)
        y = 0.26 / 2  # front surface of chest
        # x = -.553/2 to +.553/2 are the edges of the chest
        # need to add site to attach sensor and geom to generate collision to the body
        # need function for cylinders of some radius, height, and spacing (since we know its a 16x64 taxel array)
        # todo: need to scale size of geoms based on geometry since taxels size is different.

        dx = 0.553 / 32
        x = 0.553 / 2 - dx / 2

        dz = 0.514 / 30
        z = 0.13 - dz / 2

        y = 0.13

        start = [x, y, z]

        for i in range(32):  # cols
            z = start[2]
            for j in range(30):  # rows
                # logic to deal with slanted sides
                if j <= (11 / 10) * i + 19 and j <= (-11 / 10) * i + 53:
                    site_name = f"chest_{i}_{j}"
                    chest.add(
                        "geom",
                        name=site_name + "_geom",
                        type="sphere",
                        size=[0.015 / 2],
                        pos=[x, y, z],
                        rgba=self.GRAY2,
                    )
                    chest.add(
                        "site",
                        name=site_name,
                        type="sphere",
                        size=[0.015 / 2],
                        pos=[x, y, z],
                        rgba=self.GRAY2,
                    )

                    # add sensor to this site
                    self.mjcf_model.sensor.add("touch",
                                               name=f"chest_{i}_{j}_touch",
                                               site=site_name)

                z -= dz
            x -= dx

        return chest

    def createShoulders(self, chest):
        right_shoulder = chest.add(
            "body",
            name="right_shoulder",
            pos=[0.250 + 0.254 / 2 + 0.02, 0, 0],
            euler=[22.5 * 3, 0, 0],
        )

        # add geom
        # right_shoulder.add(
        #     "geom",
        #     name="right_shoulder",
        #     pos=[0, 0, 0],
        #     type="box",
        #     size=[0.254 / 2, 0.254 / 2, 0.254 / 2],
        #     rgba=self.BLACK,
        # )

        right_shoulder.add(
            "geom",
            name="right_shoulder",
            type="mesh",
            mesh="SimpleShoulderMesh",
            material="silver",
        )

        # add inertial properties
        right_shoulder.add(
            "inertial",
            mass=0.0136,
            diaginertia=[8.497e-4, 8.497e-4, 1.6992e-3],
            pos=[0, 0, 0],
        )

        # right_shoulder.add(
        #     "joint",
        #     name="right_shoulder",
        #     type="hinge",
        #     axis=[1, 0, 0],
        # )

        left_shoulder = chest.add(
            "body",
            name="left_shoulder",
            pos=[-(0.250 + 0.254 / 2 + 0.02), 0, 0],
            euler=[22.5 * 3, 0, 0],
        )

        # add geom
        # left_shoulder.add(
        #     "geom",
        #     name="left_shoulder",
        #     pos=[0, 0, 0],
        #     type="box",
        #     size=[0.254 / 2, 0.254 / 2, 0.254 / 2],
        #     rgba=self.BLACK,
        # )

        left_shoulder.add(
            "geom",
            name="left_shoulder",
            type="mesh",
            mesh="SimpleShoulderMesh",
            material="silver",
            euler=[0, 0, 180],
        )

        # add inertial properties
        left_shoulder.add(
            "inertial",
            mass=0.0136,
            diaginertia=[8.497e-4, 8.497e-4, 1.6992e-3],
            pos=[0, 0, 0],
        )

        # left_shoulder.add(
        #     "joint",
        #     name="left_shoulder",
        #     type="hinge",
        #     axis=[1, 0, 0],
        # )

        return right_shoulder, left_shoulder

    def setCompiler(self):
        self.mjcf_model.compiler.angle = "degree"

    def setOptions(self):
        self.mjcf_model.option.set_attributes(
            timestep=0.01,
            integrator='implicitfast',  #recommended by mujoco docs as best
            solver="Newton",
            jacobian="sparse",
            cone="elliptic",
        )

        self.mjcf_model.option.flag.set_attributes(gravity="enable")

    def setSimSize(self):
        self.mjcf_model.size.set_attributes(njmax=5000,
                                            nconmax=5000,
                                            nstack=5000000)

    def setVisual(self):
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
        mesh_dir = (
            "/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/meshes/"
        )

        for file in os.listdir(mesh_dir):
            if file.endswith(".stl"):
                self.mjcf_model.asset.add(
                    "mesh",
                    name=file.split(".")[0],
                    file=mesh_dir + file,
                )
                # print(file.split(".")[0])

        # add materials to define colors for different parts
        self.mjcf_model.asset.add(
            "material",
            name="vention_blue",
            rgba=self.VENTION_BLUE,
            reflectance=1,
        )
        self.mjcf_model.asset.add(
            "material",
            name="matte_black",
            rgba=self.BLACK,
            reflectance=0.2,
        )
        self.mjcf_model.asset.add(
            "material",
            name="cream",
            rgba="0.9 0.9 0.9 1",
            reflectance=0.2,
        )
        self.mjcf_model.asset.add(
            "material",
            name="silver",
            rgba="0.8 0.8 0.8 1",
            reflectance=0.8,
        )
        self.mjcf_model.asset.add(
            "material",
            name="red",
            rgba="0.8 0.2 0.2 1",
            reflectance=0.8,
        )
        self.mjcf_model.asset.add(
            "material",
            name="lcd_blue",
            rgba="0.2 0.2 0.8 1",
            reflectance=0.8,
        )
        self.mjcf_model.asset.add(
            "material",
            name="green",
            rgba="0.2 0.8 0.2 1",
            reflectance=0.8,
        )

    def setDefaults(self):
        # bellows joint limits
        eight_limit = 90 / self.num_disks
        four_limit = 110 / self.num_disks

        joint_radius8 = 0.125
        joint_radius4 = 0.08

        eight_lumped_stiffness = 136  # Nm/rad
        four_lumped_stiffness = 27  # Nm/rad
        eight_lumped_damping = 7.5  #
        four_lumped_damping = 2

        # lumped stiffness/damping uniformly distributed over each disk
        # These are spings/dampers in series, so k_total = k_disk/num_disks
        eight_stiffness = eight_lumped_stiffness * self.num_disks
        eight_damping = eight_lumped_damping * self.num_disks
        four_stiffness = four_lumped_stiffness * self.num_disks
        four_damping = four_lumped_damping * self.num_disks

        # create default class for 8 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
        disk_class = self.mjcf_model.default.add("default",
                                                 dclass="large_joint")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[joint_radius8, self.disk_half_height * 0.8],
        )
        disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            stiffness=eight_stiffness,
            damping=eight_damping,
            pos=[0, 0, self.disk_height],
            limited="true",
            range=[-eight_limit, eight_limit],
        )

        # create default class for 8 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
        disk_class = self.mjcf_model.default.add("default",
                                                 dclass="medium_joint")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[joint_radius4, self.disk_half_height * 0.8],
        )
        disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            stiffness=four_stiffness,
            damping=four_damping,
            pos=[0, 0, self.disk_height],
            limited="true",
            range=[-four_limit, four_limit],
        )

        disk_class = self.mjcf_model.default.add("default",
                                                 dclass="small_joint")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[joint_radius4, self.disk_half_height * 0.8],
        )
        disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            stiffness=four_stiffness,
            damping=four_damping,
            pos=[0, 0, self.disk_height],
            limited="true",
            range=[-four_limit, four_limit],
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    torso = Baloo(
        "baloo_torso",
        5,
    )

    # print(torso.mjcf_model)
    # export_with_assets(torso.mjcf_model,'.', "baloo.xml", zero_threshold=1e-8)

    xml = torso.mjcf_model.to_xml_string()

    # bandaid for weird bug to replace strings inserted after file names:
    # remove random letters and numbers in between dash and .stl from comments above
    xml = re.sub(r"-(.*?).stl", ".stl", xml)

    # prepend absolute path to all stl in xml file
    xml = re.sub(
        r"file=\"",
        'file="/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/meshes/',
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
    # context = mujoco.GLContext(1000,1000)
    # context.make_current()
    # fig = mujoco.MjvFigure()
    # viewport = mujoco.MjrRect(0, 0, 800, 800)
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

            # print(data.sensor('vive_tracker').data)
            # print(data.sensor('0_B0_framequat').id)
            # print(data.sensordata[:4])
            # print(f"First Disk Quat: {data.sensor('0_B0_framequat').data}")

            # print(data.sensor("0_B4_framequat").id)
            # print(data.sensordata[4:8])
            # print(f"Last Disk Quat: {data.sensor('0_B4_framequat').data}")

            # #conjugate and multiply
            # # R = mujoco.mju_quat2Mat(data.sensor('0_B4_framequat').data)

            # print(data.sensordata[8:11])
            # print(f"Last Disk Vel: {data.sensor('0_B4_frameangvel').data}")

            # print(data.sensordata[-4:])
            # print(f"Vive Tracker: {data.sensor('vive_tracker').data}")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            plot_start = time.time()
            plotter.update(data)
            # print(time.time() - plot_start)

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
