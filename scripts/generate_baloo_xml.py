#TODO: maybe add params to include sensor config, and if you want manipuland

import os
import re
from dm_control import mjcf
import numpy as np

from regex import F
import yaml
from scipy.spatial.transform import Rotation as R
import mujoco

from importlib.resources import path


class Baloo:
    def __init__(self, name, asset_dir: str) -> None:
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)
        print("Setting up the model...")
        self._setupModel(asset_dir)

        if self.manipuland != 'None':
            print("Creating manipuland...")
            self._createManipuland()

        print("Creating base...")
        linear_actuator = self._createBase(self.mjcf_model.worldbody)
        print("Creating chest...")
        chest = self.createChest(linear_actuator)
        print("Creating shoulders...")
        right_shoulder, left_shoulder = self.createShoulders(chest)

        print("Building right arm...")
        right_last_disk = self._buildLargeJoint(right_shoulder, "right")
        right_link0 = self.addLink0(right_last_disk, "right")
        last_disk = self._buildMediumJoint(right_link0, "right")
        right_link1 = self.addLink1(last_disk, "right")
        right_ee_disk = self._buildSmallJoint(right_link1, "right")

        print("Building left arm...")
        left_last_disk = self._buildLargeJoint(left_shoulder, "left")
        left_link0 = self.addLink0(left_last_disk, "left")
        last_disk = self._buildMediumJoint(left_link0, "left")
        left_link1 = self.addLink1(last_disk, "left")
        left_ee_disk = self._buildSmallJoint(left_link1, "left")

        # # exclude contact between chest and each link on each arm
        # #exclude contacts between left and right arm links
        # self._setContactDetection()

        # comment out if not whole arm is built (i.e. for testing)
        print("Adding sensors to right arm...")
        self._addSensors("right")
        print("Adding sensors to left arm...")
        self._addSensors("left")

        print(f"{name} model building completed.")

        # # add mocap bodies to end effector disks, can't really do externally...
        # left_ee_mocap = self.mjcf_model.worldbody.add(
        #     "body",
        #     name="left_ee_mocap",
        #     mocap="true",
        #     pos=[0, 0, 0],
        # )

        # left_ee_mocap.add(
        #     "geom",
        #     name="left_ee_mocap",
        #     type="box",
        #     size=[0.05] * 3,
        #     rgba=[1, 0, 0, 1],
        #     contype=0,
        #     conaffinity=0,
        # )

        # self.mjcf_model.worldbody.add(
        #     "body",
        #     name="left_ee_mocap",
        #     mocap="true",
        #     pos=[0, 0, 0],
        # )

    def _loadPlugins(self):
        print(
            f"Remember to build all plugins with current version of mujoco (v {mujoco.__version__}) before running this script."
        )
        plugin = self.mjcf_model.extension.add(
            "plugin",
            plugin="mujoco.sensor.joint_angle_estimator",
        )

        elevator_plugin = self.mjcf_model.extension.add(
            "plugin",
            plugin="mujoco.actuator.motion_profile_servo",
        )

    def _setupModel(self, asset_dir):

        self.ORANGE = [0.8, 0.2, 0.1, 1]
        self.VENTION_BLUE = [0, 40 / 255, 80 / 255, 1]
        self.BLACK = [0 / 255, 0 / 255, 0 / 255, 1]
        self.WHITE = [255, 255, 255, 1]
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        self.GRAY2 = [50 / 255, 50 / 255, 50 / 255, 0.7]
        # self.ORANGE = [15 / 256.0, 10 / 256.0, 222 / 256.0, 1]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self._setCompiler()
        self._setOptions()
        self._setSimSize()
        self._setVisual()
        self._addAssets(asset_dir)
        self._loadParams(asset_dir)
        self._setCustomData()
        self._setDefaults()
        # create world plane
        self.mjcf_model.worldbody.add(
            "geom",
            condim=1,
            material="groundplane",
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
        self._loadPlugins()

    def _loadParams(self, asset_dir):
        with open(os.path.join(asset_dir, "params.yaml"), "r") as file:
            params = yaml.safe_load(file)

        self.small_joint_radius = params["small_joint"]["radius"]
        self.small_joint_mass = params["small_joint"]["mass"]
        self.small_joint_bellows_radius = params["small_joint"][
            "bellows_radius"]
        self.small_joint_bend_limit = params["small_joint"]["bend_limit"]
        self.small_joint_lumped_stiffness = params["small_joint"][
            "lumped_stiffness"]
        self.small_joint_lumped_damping = params["small_joint"][
            "lumped_damping"]
        self.small_joint_area = params["small_joint"]["bellows_effective_area"]

        self.medium_joint_radius = params["medium_joint"]["radius"]
        self.medium_joint_mass = params["medium_joint"]["mass"]
        self.medium_joint_bellows_radius = params["medium_joint"][
            "bellows_radius"]
        self.medium_joint_bend_limit = params["medium_joint"]["bend_limit"]
        self.medium_joint_lumped_stiffness = params["medium_joint"][
            "lumped_stiffness"]
        self.medium_joint_lumped_damping = params["medium_joint"][
            "lumped_damping"]
        self.medium_joint_area = params["medium_joint"][
            "bellows_effective_area"]

        self.large_joint_radius = params["large_joint"]["radius"]
        self.large_joint_mass = params["large_joint"]["mass"]
        self.large_joint_bellows_radius = params["large_joint"][
            "bellows_radius"]
        self.large_joint_bend_limit = params["large_joint"]["bend_limit"]
        self.large_joint_lumped_stiffness = params["large_joint"][
            "lumped_stiffness"]
        self.large_joint_lumped_damping = params["large_joint"][
            "lumped_damping"]
        self.large_joint_area = params["large_joint"]["bellows_effective_area"]

        self.link0_height = params["link0"]["height"]
        self.link0_radius = params["link0"]["radius"]
        self.link0_mass = params["link0"]["mass"]

        self.link1_height = params["link1"]["height"]
        self.link1_radius = params["link1"]["radius"]
        self.link1_mass = params["link1"]["mass"]

        # some joint measurements common (hopefully) among all joints
        self.joint_height = params["general"]["joint_height"]
        self.pmax = params["general"]["max_pressure"]

        self.bellows_areas = [
            self.large_joint_area, self.medium_joint_area,
            self.small_joint_area
        ]

        self.pressure_time_consts = [
            params["large_joint"]["pressure_time_constant"],
            params["medium_joint"]["pressure_time_constant"],
            params["small_joint"]["pressure_time_constant"]
        ]

        self.arm_angle = params["general"]["arm_angle"]

        self.num_disks = params["general"]["num_disks"]
        num_spaces = self.num_disks - 1
        self.num_universal_joints = num_spaces
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2

        self.manipuland = params["general"]["manipuland"]

        assert self.manipuland in ['box', 'None'
                                   ], "manipuland must be 'box' or 'None'"
        self.useTactileSensors = params["general"]["tactile_sensors"]

    def _setContactDetection(self):
        self.mjcf_model.contact.add(
            "exclude",
            name="chest_left_link0",
            body1="chest",
            body2="left_link0",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="chest_left_link1",
            body1="chest",
            body2="left_link1",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="chest_right_link0",
            body1="chest",
            body2="right_link0",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="chest_right_link1",
            body1="chest",
            body2="right_link1",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="left_link0_right_link0",
            body1="left_link0",
            body2="right_link0",
        )

        self.mjcf_model.contact.add(
            "exclude",
            name="left_link1_right_link1",
            body1="left_link1",
            body2="right_link1",
        )

    def _createManipuland(self):
        # pos = (-.5, .5)m, box of .5 m side,
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
            type="box",
            size=[width / 2, depth / 2, height / 2],
            rgba=self.ORANGE,
            contype=2,
            conaffinity=1,
        )

        #contype and affinity are set up so taxels only respond to being touched by the manipuland, not by other bodies.

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

    def _setCustomData(self):
        self.mjcf_model.custom.add(
            "numeric",
            name="num_disks",
            size=1,
            data=[self.num_disks],
        )

    def _setContacts(self):
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

    def _addActuators(self, side, joint_num):

        for i in range(4):
            if joint_num == 0:
                #large joint so these need to be invisible
                bellows = self.mjcf_model.tendon.add(
                    "spatial",
                    name=f"{side}_j{joint_num}_bellows{i}",
                    dclass="tendon",
                    rgba=[0, 0, 0, 0])
            else:
                #add the bellows to the model
                bellows = self.mjcf_model.tendon.add(
                    "spatial",
                    name=f"{side}_j{joint_num}_bellows{i}",
                    dclass="tendon")

            #add all sites along the disks for the bellows to attach to
            for j in range(self.num_disks):
                site = self.mjcf_model.find(
                    'site', f"{side}_j{joint_num}_b{j}_site{i}")
                bellows.add("site", site=site)

        #add actuator to each built tendon
        self.mjcf_model.actuator.add(
            "cylinder",
            name=f"{side}_j{joint_num}_p0",
            tendon=f"{side}_j{joint_num}_bellows0",
            area=self.bellows_areas[joint_num] *
            1000,  # 1000 since inputs are in kPa
            dclass="cylinder",
            timeconst=self.pressure_time_consts[joint_num])

        #add actuator to side_tendon
        self.mjcf_model.actuator.add(
            "cylinder",
            name=f"{side}_j{joint_num}_p1",
            tendon=f"{side}_j{joint_num}_bellows1",
            area=self.bellows_areas[joint_num] *
            1000,  # 1000 since inputs are in kPa
            dclass='cylinder',
            timeconst=self.pressure_time_consts[joint_num])

        self.mjcf_model.actuator.add(
            "cylinder",
            name=f"{side}_j{joint_num}_p2",
            tendon=f"{side}_j{joint_num}_bellows2",
            area=self.bellows_areas[joint_num] *
            1000,  # 1000 since inputs are in kPa
            dclass='cylinder',
            timeconst=self.pressure_time_consts[joint_num])

        self.mjcf_model.actuator.add(
            "cylinder",
            name=f"{side}_j{joint_num}_p3",
            tendon=f"{side}_j{joint_num}_bellows3",
            area=self.bellows_areas[joint_num] *
            1000,  # 1000 since inputs are in kPa
            dclass='cylinder',
            timeconst=self.pressure_time_consts[joint_num])

    def _addSensors(self, side):
        # add framequat relative to world frame on base and tip of each joint
        for joint_num in range(3):
            ##### BASE #####
            self.mjcf_model.sensor.add(
                "framequat",
                name=f"{side}_j{joint_num}_B0_framequat",
                objtype="body",
                objname=f"{side}_j{joint_num}_B0",
            )

            #### TIP ####
            self.mjcf_model.sensor.add(
                "framequat",
                name=f"{side}_j{joint_num}_B{self.num_disks-1}_framequat",
                objtype="body",
                objname=f"{side}_j{joint_num}_B{self.num_disks-1}",
            )

            #add frameangvel to tip, ref to base frame
            self.mjcf_model.sensor.add(
                "frameangvel",
                name=f"{side}_j{joint_num}_B{self.num_disks-1}_frameangvel",
                objtype="body",
                objname=f"{side}_j{joint_num}_B{self.num_disks-1}",
                reftype="body",
                refname=f"{side}_j{joint_num}_B0",
            )

            self.mjcf_model.sensor.add(
                "plugin",
                plugin="mujoco.sensor.joint_angle_estimator",
                name=f'{side}_j{joint_num}')

            #add tendon length sensors to all joints to see what they are at
            for i in range(4):
                self.mjcf_model.sensor.add(
                    "tendonpos",
                    name=f"{side}_j{joint_num}_bellows{i}",
                    tendon=f"{side}_j{joint_num}_bellows{i}",
                )

    def addLink0(self, body, side):
        link = body.add(
            "body",
            name=f"{side}_link0",
            pos=[0, 0, (self.disk_half_height + self.link0_height / 2)],
            euler=[0, 0, -45],
        )

        link.add(
            "geom",
            name=f"{side}_link0",
            type="cylinder",
            size=[self.link0_radius, self.link0_height / 2],
            rgba=self.BLACK,
        )

        Ixx = self.link0_mass * (3 * self.link0_radius**2 +
                                 self.link0_height**2) / 12
        Iyy = self.link0_mass * (3 * self.link0_radius**2 +
                                 self.link0_height**2) / 12
        Izz = self.link0_mass * self.link0_radius**2 / 2

        link.add("inertial",
                 pos=[0, 0, 112e-3 - self.link0_height / 2],
                 diaginertia=[Ixx, Iyy, Izz],
                 mass=self.link0_mass)

        self.add_tactile_sleeve(side, link, 0, self.link0_height,
                                self.link0_radius)

        return link

    def add_tactile_sleeve(self, side, link, linknum, link_height, r):
        if self.useTactileSensors:
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
                        contype=0,
                        conaffinity=2,  #to exclude contact with joint disks
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
            pos=[0, 0, (self.disk_half_height + self.link1_height / 2)],
            euler=[0, 0, -45],
        )

        link.add(
            "geom",
            name=f"{side}_link1",
            type="cylinder",
            size=[self.link1_radius, self.link1_height / 2],
            rgba=self.BLACK,
        )

        Ixx = self.link1_mass * (3 * self.link1_radius**2 +
                                 self.link1_height**2) / 12
        Iyy = self.link1_mass * (3 * self.link1_radius**2 +
                                 self.link1_height**2) / 12
        Izz = self.link1_mass * self.link1_radius**2 / 2

        link.add(
            "inertial",
            pos=[0, 0, 87e-3 - self.link1_height / 2],
            diaginertia=[Ixx, Iyy, Izz],
            mass=self.link1_mass,
        )

        self.add_tactile_sleeve(side, link, 1, self.link1_height,
                                self.link1_radius)

        return link

    def _buildLargeJoint(self, parent_body, side):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        disk_mass = self.large_joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically).
        Ixy = (disk_mass *
               (3 * self.large_joint_radius**2 + self.disk_height**2)) / 12
        Iz = (disk_mass * self.large_joint_radius**2) / 2

        joint_num = 0

        # create first body, whose frame is offset
        first_disk = parent_body.add(
            "body",
            name=f"{side}_j{joint_num}_B0",
            # childclass="large_joint",
            pos=[0, 0, -(185e-3 + self.disk_half_height)],
            euler=[180, 0, -45],
        )
        first_disk.add(
            "geom",
            name=f"{side}_j{joint_num}_G0",
            dclass="large_joint",
        )

        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        self._addFourSitesToDisk(first_disk, side, joint_num, 0, "large")
        self._addEightSitesToDisk(first_disk, side, joint_num, 0)

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_j{joint_num}_B{i}",
                pos=[0, 0, (2 * self.disk_height)],
                # childclass="large_joint",
            )
            body.add(
                "geom",
                name=f"{side}_j{joint_num}_G{i}",
                dclass="large_joint",
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])
            body.add("joint",
                     name=f"{side}_j{joint_num}_Jx_{i-1}",
                     dclass="large_joint",
                     axis=self.X)
            body.add("joint",
                     name=f"{side}_j{joint_num}_Jy_{i-1}",
                     dclass="large_joint",
                     axis=self.Y)

            self._addFourSitesToDisk(body, side, joint_num, i, "large")
            self._addEightSitesToDisk(body, side, joint_num, i)
            prev_body = body

        self._addActuators(side, joint_num)
        self._addEightVizTendons(side)
        return body

    def _addEightSitesToDisk(self, disk, side, joint_num, disk_num):
        Rplus = R.from_euler('z', 22.5, degrees=True)
        Rminus = R.from_euler('z', -22.5, degrees=True)
        #this is only applicable to the 8 bellows

        for i in range(4):
            site = self.mjcf_model.find(
                'site', f"{side}_j{joint_num}_b{disk_num}_site{i}")
            pos = np.array(site.pos)
            pos_plus = Rplus.apply(pos)
            pos_minus = Rminus.apply(pos)

            disk.add(
                "site",
                name=f"{side}_j{joint_num}_b{disk_num}_site{i}_A",
                pos=pos_plus,
                rgba=[1, 1, 1, 1],
            )

            disk.add(
                "site",
                name=f"{side}_j{joint_num}_b{disk_num}_site{i}_B",
                pos=pos_minus,
                rgba=[1, 1, 1, 1],
                group=0,
            )

    def _addEightVizTendons(self, side):
        joint_num = 0
        for i in range(4):
            bellowsA = self.mjcf_model.tendon.add(
                "spatial",
                name=f"{side}_j{joint_num}_bellows{i}_A",
                dclass="tendon",
            )
            bellowsB = self.mjcf_model.tendon.add(
                "spatial",
                name=f"{side}_j{joint_num}_bellows{i}_B",
                dclass="tendon",
            )

            for j in range(self.num_disks):
                siteA = self.mjcf_model.find(
                    'site', f"{side}_j{joint_num}_b{j}_site{i}_A")
                bellowsA.add("site", site=siteA)

                siteB = self.mjcf_model.find(
                    'site', f"{side}_j{joint_num}_b{j}_site{i}_B")
                bellowsB.add("site", site=siteB)

    def _buildMediumJoint(self, body, side):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        disk_mass = self.medium_joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass *
               (3 * self.medium_joint_radius**2 + self.disk_height**2)) / 12
        Iz = (disk_mass * self.medium_joint_radius**2) / 2

        joint_num = 1

        # create first body, whose frame is offset
        first_disk = body.add(
            "body",
            name=f"{side}_j{joint_num}_B0",
            # childclass="medium_joint",
            pos=[0, 0, (self.link0_height / 2 + self.disk_half_height)
                 ],  # from pneubotics
            euler=[0, 0, 45],
        )
        first_disk.add(
            "geom",
            name=f"{side}_j{joint_num}_G0",
            dclass='medium_joint',
        )
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        self._addFourSitesToDisk(first_disk, side, joint_num, 0, "medium")

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom, add joints
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_j{joint_num}_B{i}",
                pos=[0, 0, (2 * self.disk_height)],
            )
            body.add(
                "geom",
                name=f"{side}_j{joint_num}_G{i}",
                dclass='medium_joint',
            )
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])

            #creates motion dof between body and the body's parent (i.e. prev_body)
            body.add("joint",
                     name=f"{side}_j{joint_num}_Jx_{i-1}",
                     axis=self.X,
                     dclass='medium_joint')
            body.add("joint",
                     name=f"{side}_j{joint_num}_Jy_{i-1}",
                     axis=self.Y,
                     dclass='medium_joint')

            self._addFourSitesToDisk(body, side, joint_num, i, "medium")
            prev_body = body

        self._addActuators(side, joint_num)
        return body

    def _addFourSitesToDisk(self, disk, side, joint_num, disk_num, size):
        if size == "large":
            loc = self.large_joint_bellows_radius
        elif size == "medium":
            loc = self.medium_joint_bellows_radius
        else:
            loc = self.small_joint_bellows_radius

        # chamber 0 is +y
        disk.add(
            "site",
            name=f"{side}_j{joint_num}_b{disk_num}_site{0}",
            pos=[0, loc, 0],
            dclass="bellows_site",
        )
        disk.add(
            "site",
            name=f"{side}_j{joint_num}_b{disk_num}_site{1}",
            pos=[0, -loc, 0],
            dclass="bellows_site",
        )
        # chamber 2 is -x
        disk.add(
            "site",
            name=f"{side}_j{joint_num}_b{disk_num}_site{2}",
            pos=[-loc, 0, 0],
            dclass="bellows_site",
        )

        # chamber 3 is +x
        disk.add(
            "site",
            name=f"{side}_j{joint_num}_b{disk_num}_site{3}",
            pos=[loc, 0, 0],
            dclass="bellows_site",
        )

    def _buildSmallJoint(self, body, side):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        disk_mass = self.small_joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass *
               (3 * self.small_joint_radius**2 + self.disk_height**2)) / 12
        Iz = (disk_mass * self.small_joint_radius**2) / 2
        # create first body, whose frame is offset
        joint_num = 2

        first_disk = body.add(
            "body",
            name=f"{side}_j{joint_num}_B0",
            # childclass="small_joint",
            pos=[0, 0, (.08 + self.disk_half_height)],
            euler=[0, 0, 45],
        )
        first_disk.add("geom",
                       name=f"{side}_j{joint_num}_G0",
                       dclass='small_joint')
        first_disk.add("inertial",
                       mass=disk_mass,
                       diaginertia=[Ixy, Ixy, Iz],
                       pos=[0, 0, 0])

        self._addFourSitesToDisk(first_disk,
                                 side,
                                 joint_num,
                                 disk_num=0,
                                 size="small")

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{side}_j{joint_num}_B{i}",
                pos=[0, 0, (2 * self.disk_height)],
                # childclass="small_joint",
            )
            body.add("geom",
                     name=f"{side}_j{joint_num}_G{i}",
                     dclass='small_joint')
            body.add("inertial",
                     mass=disk_mass,
                     diaginertia=[Ixy, Ixy, Iz],
                     pos=[0, 0, 0])

            body.add("joint",
                     name=f"{side}_j{joint_num}_Jx_{i-1}",
                     dclass='small_joint',
                     axis=self.X)
            body.add("joint",
                     name=f"{side}_j{joint_num}_Jy_{i-1}",
                     dclass='small_joint',
                     axis=self.Y)

            self._addFourSitesToDisk(body, side, joint_num, i, "small")

            prev_body = body

        self._addActuators(side, joint_num)
        return body

    def _createBase(self, body):
        # create linear actuator and torso from which to hang arms
        base = body.add("body", name="base", pos=[0, 0, 0], euler=[0, 0, 0])

        base.add(
            "geom",
            type="mesh",
            mesh="LeftBaseMesh",
            material="vention_blue",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="RightBaseMesh",
            material="vention_blue",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="BaseFrameMesh",
            material="vention_blue",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="LinearActuatorMesh",
            material="vention_blue",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="PneumaticInletMesh",
            material="silver",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="PowerButtonMesh",
            material="red",
        )

        base.add(
            "geom",
            type="mesh",
            mesh="ControlBoxMesh",
            material="matte_black",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="EstopPlugMesh",
            material="green",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="EthernetJackMesh",
            material="silver",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="LCDScreenMesh",
            material="lcd_blue",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="LeftBackWheelFootMesh",
            material="matte_black",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="LeftFrontWheelFootMesh",
            material="matte_black",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="RightBackWheelFootMesh",
            material="matte_black",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="RightFrontWheelFootMesh",
            material="matte_black",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="LeftBackWheelMesh",
            material="cream",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="LeftFrontWheelMesh",
            material="cream",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="RightBackWheelMesh",
            material="cream",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="RightFrontWheelMesh",
            material="cream",
        )
        base.add(
            "geom",
            type="mesh",
            mesh="StepperMesh",
            material="matte_black",
        )
        return base

    def createChest(self, linear_actuator):
        chest = linear_actuator.add("body",
                                    name="chest",
                                    pos=[0, 195e-3, 1554e-3],
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

        elevator_plugin = self.mjcf_model.actuator.add(
            'plugin',
            name='elevator',
            plugin="mujoco.actuator.motion_profile_servo",
            ctrllimited=True,
            ctrlrange=[-1000, 0],
            joint="linear_actuator",
        )

        elevator_plugin.add(
            "config",
            key="kp",
            value="0.3",  #m/s per meter
        )

        elevator_plugin.add(
            "config",
            key="kv",
            value="1000",
        )

        elevator_plugin.add(
            "config",
            key="zeta",
            value="1",
        )

        elevator_plugin.add(
            "config",
            key="omega_n",
            value="0.3",
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

        if self.useTactileSensors:
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
            pos=[425e-3, 0, 0],
            euler=[self.arm_angle, 0, 0],
        )

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
            pos=[-425e-3, 0, 0],
            euler=[self.arm_angle, 0, 0],
        )

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

    def _setCompiler(self):
        self.mjcf_model.compiler.angle = "degree"

    def _setOptions(self):
        self.mjcf_model.option.set_attributes(
            timestep=0.005,
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

    def _addAssets(self, asset_dir):
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
            name="groundplane",
            texture="texplane",
            texuniform="true",
            # reflectance=0.1,
        )

        # add assets for all the meshes in the meshes directory
        mesh_dir = os.path.join(asset_dir, "meshes")
        for file in os.listdir(mesh_dir):
            if file.endswith(".stl"):
                self.mjcf_model.asset.add(
                    "mesh",
                    name=file.split(".")[0],
                    file=os.path.join(mesh_dir, file),
                )

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

    def _setDefaults(self):
        # lumped stiffness/damping uniformly distributed over each disk
        # These are spings/dampers in series, so k_total = k_disk/num_disks
        large_stiffness = self.large_joint_lumped_stiffness * self.num_universal_joints
        large_damping = self.large_joint_lumped_damping * self.num_universal_joints
        medium_stiffness = self.medium_joint_lumped_stiffness * self.num_universal_joints
        medium_damping = self.medium_joint_lumped_damping * self.num_universal_joints
        small_stiffness = self.small_joint_lumped_stiffness * self.num_universal_joints
        small_damping = self.small_joint_lumped_damping * self.num_universal_joints

        # create default class for 8 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
        large_class = self.mjcf_model.default.add("default",
                                                  dclass="large_joint")
        large_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[self.large_joint_radius, self.disk_half_height],
        )
        large_class.joint.set_attributes(
            type="hinge",
            group=0,
            stiffness=large_stiffness,
            damping=large_damping,
            pos=[0, 0, -self.disk_height],
            # limited="true",
            # range=[-eight_limit, eight_limit], #todo: need limits? or just springs?
        )

        # create default class for 8 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
        medium_class = self.mjcf_model.default.add("default",
                                                   dclass="medium_joint")
        medium_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[self.medium_joint_radius, self.disk_half_height],
        )
        medium_class.joint.set_attributes(
            type="hinge",
            group=0,
            stiffness=medium_stiffness,
            damping=medium_damping,
            pos=[0, 0, -self.disk_height],
            # limited="true",
            # range=[-four_limit, four_limit],
        )

        small_class = self.mjcf_model.default.add("default",
                                                  dclass="small_joint")
        small_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[self.small_joint_radius, self.disk_half_height],
        )
        small_class.joint.set_attributes(
            type="hinge",
            group=0,
            stiffness=small_stiffness,
            damping=small_damping,
            pos=[0, 0, -self.disk_height],
            # limited="true",
            # range=[-four_limit, four_limit],
        )

        bellows_site_class = self.mjcf_model.default.add("default",
                                                         dclass="bellows_site")
        bellows_site_class.site.set_attributes(rgba=[0, 1, 0, 1])

        tendon_class = self.mjcf_model.default.add("default", dclass="tendon")

        tendon_class.tendon.set_attributes(width=0.03)

        cylinder_class = self.mjcf_model.default.add("default",
                                                     dclass="cylinder")

        cylinder_class.cylinder.set_attributes(
            ctrllimited="true",
            ctrlrange=[0, self.pmax / 1000],  #kpa
        )

    def to_clean_xml_string(self, asset_dir=str):
        '''
        pymjcf does some weird stuff. This function cleans up those weird things to 
        avoid loading problems later. 
        '''
        xml = self.mjcf_model.to_xml_string()
        # bandaid for weird bug to replace strings inserted after file names:
        # remove random letters and numbers in between dash and .stl from comments above
        xml = re.sub(r"-(.*?).stl", ".stl", xml)

        # prepend absolute path to all stl in xml file
        mesh_dir = os.path.join(asset_dir, "meshes")
        xml = re.sub(
            r"file=\"",
            f'file="{mesh_dir}/',
            xml,
        )
        return xml


def generate_xml():
    np.set_printoptions(precision=3, suppress=True)

    #get version of package
    from pathlib import Path as path

    project_root = path(__file__).resolve().parent.parent

    import toml
    toml_path = project_root / "pyproject.toml"

    with open(toml_path) as f:
        data = toml.load(f)
    ver = data["project"]["version"]

    torso = Baloo(f"baloo_v{ver}",
                  asset_dir=project_root / "baloo_mujoco_sim" / "assets")

    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    with open(
            project_root / "baloo_mujoco_sim" / "assets" / f"baloo_v{ver}.xml",
            "w") as f:
        f.write(
            torso.to_clean_xml_string(asset_dir=project_root /
                                      "baloo_mujoco_sim" / "assets"))
    f.close()


if __name__ == "__main__":
    generate_xml()
#     np.set_printoptions(precision=3, suppress=True)

#     from importlib.metadata import version
#     ver = version(baloo_mujoco_sim.__name__)

#     torso = Baloo(f"baloo_v{ver}")

#     # to actually write xml file. There's a weird bug in the stl that you need to fix.
#     with open(
#             os.path.join(baloo_mujoco_sim.__path__[0],
#                          f"assets/baloo_v{ver}.xml"), "w") as f:
#         f.write(torso.to_clean_xml_string())

#     f.close()
