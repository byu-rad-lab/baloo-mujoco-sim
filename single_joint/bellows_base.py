"""
Base class for all bellows continuum joints. All joints have 4 pressure chambers. All have 4 control inputs.
"""

from dm_control import mjcf
from abc import ABC, abstractmethod


class BellowsBase():
    def __init__(self, name, num_disks, joint_height, joint_radius,
                 joint_mass):
        self.name = name
        self.mjcf_model = mjcf.RootElement(model=name)
        self.num_disks = num_disks
        self.joint_height = joint_height
        self.joint_radius = joint_radius
        self.joint_mass = joint_mass

        num_spaces = self.num_disks - 1
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2
        self.disk_mass = joint_mass / self.num_disks

        # get moment of inertia of each disk (as thin cylinder)
        self.Ixy = (self.disk_mass *
                    (3 * joint_radius**2 + self.disk_height**2)) / 12
        self.Iz = (self.disk_mass * joint_radius**2) / 2

        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]
        self.ORANGE = [0.8, 0.2, 0.1, 1]
        self.VENTION_BLUE = [0, 40 / 255, 80 / 255, 1]
        self.BLACK = [0 / 255, 0 / 255, 0 / 255, 1]
        self.WHITE = [255, 255, 255, 1]
        self.GRAY = [120 / 255, 120 / 255, 120 / 255, 0.7]
        self.GRAY2 = [50 / 255, 50 / 255, 50 / 255, 0.7]

        self.setDefaults()
        self.last_disk = self._buildContinuumJoint(self.mjcf_model.worldbody)

        for i in range(4):
            self._addBellow(i)
            self._addActuator(i)

    def __repr__(self) -> str:
        return self.mjcf_model.to_xml_string()

    def _buildContinuumJoint(self, parent_body):
        # total joint -> [disk,space,disk,....,space,disk]

        # create first body, whose frame is offset
        disk0 = self._buildDisk(parent_body, 0)

        prev_disk = disk0

        # +1 bc I already made first disk above)
        for i in range(1, self.num_disks):
            disk = self._buildDisk(prev_disk, i)
            self._addJoint2Disk(disk, i)
            prev_disk = disk

        return disk

    def _buildDisk(self, parent_body, disk_num):
        if disk_num == 0:
            #frame is slightly offset
            disk = parent_body.add(
                "body",
                name=f"{self.name}_body{disk_num}",
                childclass="small_joint",
                pos=[0, 0, self.disk_half_height],
                euler=[0, 0, 0],
            )

        else:
            disk = parent_body.add(
                "body",
                name=f"{self.name}_body{disk_num}",
                pos=[0, 0, (2 * self.disk_height)],
            )

        disk.add(
            "geom",
            name=f"{self.name}_geom{disk_num}",
            rgba=[.3, .3, .3, 1],
            group=1,
        )

        disk.add("inertial",
                 mass=self.disk_mass,
                 diaginertia=[self.Ixy, self.Ixy, self.Iz],
                 pos=[0, 0, 0])

        disk.add(
            "site",
            name=f"{self.name}_site{disk_num}_chamber2",
            pos=[self.joint_radius / 1.8, 0, 0],
            rgba=[0, 1, 0, 1],
            group=2,
        )

        disk.add(
            "site",
            name=f"{self.name}_site{disk_num}_chamber3",
            pos=[-self.joint_radius / 1.8, 0, 0],
            rgba=[0, 1, 0, 1],
            group=3,
        )

        disk.add(
            "site",
            name=f"{self.name}_site{disk_num}_chamber0",
            pos=[0, self.joint_radius / 1.8, 0],
            rgba=[0, 1, 0, 1],
            group=0,
        )

        disk.add(
            "site",
            name=f"{self.name}_site{disk_num}_chamber1",
            pos=[0, -self.joint_radius / 1.8, 0],
            rgba=[0, 1, 0, 1],
            group=1,
        )

        return disk

    def _addJoint2Disk(self, body, disk_num):
        '''
        a joint creates motion degrees of freedom between the body where it is defined and the body’s parent.
        '''
        # {self.name}_site{disk_num}_chamber2
        body.add("joint",
                 name=f"{self.name}_jointX{disk_num-1}",
                 axis=self.X,
                 pos=[0, 0, -self.disk_height])

        body.add("joint",
                 name=f"{self.name}_jointY{disk_num-1}",
                 axis=self.Y,
                 pos=[0, 0, -self.disk_height])

        body.add(
            "site",
            name=f"{self.name}_site{disk_num}_rot",
            pos=[0, 0, -self.disk_height],
            rgba=[1, 0, 0, 1],
            group=4,
        )

    def _addActuator(self, bellows_num):
        self.mjcf_model.actuator.add("cylinder",
                                     name=f"{self.name}_p{bellows_num}",
                                     tendon=f"{self.name}_tendon{bellows_num}",
                                     diameter=0.05,
                                     ctrllimited=True,
                                     ctrlrange=[0, 1000],
                                     gear=[0.5 * 1000],
                                     timeconst=0.2)

    def _addBellow(self, bellows_num):
        tendon = self.mjcf_model.tendon.add(
            "spatial",
            name=f"{self.name}_tendon{bellows_num}",
            width=0.03,
            rgba=[.6, .6, .6, 1])

        for i in range(self.num_disks):
            tendon.add("site",
                       site=f"{self.name}_site{i}_chamber{bellows_num}")

    def setDefaults(self):
        #?should I put springs and dampers on joints? or on tendons?
        #! I feel like it makes the most sense to do on tendons now...
        #! mujoco recommends damping be put on joints
        #! joint stiffness or tendon stiffness?
        # bellows joint limits
        four_limit = 110 / self.num_disks

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

        disk_class = self.mjcf_model.default.add("default",
                                                 dclass="small_joint")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.GRAY,
            size=[self.joint_radius, self.disk_half_height],
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

    def get_connection_site(self):
        site = self.last_disk.add(
            "site",
            name=f"{self.name}_connection_site",
            pos=[0, 0, self.disk_half_height],
            rgba=[1, 1, 0, 1],
        )
        return site
