import time
from dm_control import mjcf
from warnings import warn
import mujoco.viewer
from bellows_base import BellowsBase
import numpy as np
from scipy.spatial.transform import Rotation as R


class FourBellows(BellowsBase):
    def __init__(self, name, num_disks, joint_height, joint_radius,
                 joint_mass) -> None:

        super().__init__(name, num_disks, joint_height, joint_radius,
                         joint_mass)


class EightBellows(BellowsBase):
    def __init__(self, name, num_disks, joint_height, joint_radius,
                 joint_mass) -> None:

        super().__init__(name, num_disks, joint_height, joint_radius,
                         joint_mass)

        self._add_visual_tendons()

    def _add_visual_tendons(self):
        Rplus30 = R.from_euler('z', 22.5, degrees=True)
        Rminus30 = R.from_euler('z', -22.5, degrees=True)

        # add 8 sites to each disk for visual tendons to attach to
        for disk_num in range(self.num_disks):
            disk = self.mjcf_model.find('body', f"{self.name}_body{disk_num}")

            for j in range(4):
                site = self.mjcf_model.find(
                    'site', f"{self.name}_site{disk_num}_chamber{j}")
                pos = np.array(site.pos)

                #rotate pos by +30 and -30 degres to get 2 sites to either side of axis
                pos_plus30 = Rplus30.apply(pos)
                pos_minus30 = Rminus30.apply(pos)
                # print(j)
                #add site to disk
                disk.add(
                    "site",
                    name=f"{self.name}_site{disk_num}_chamber{j}_plus30",
                    pos=pos_plus30,
                    rgba=[1, 1, 1, 1],
                    group=1,
                )
                disk.add(
                    "site",
                    name=f"{self.name}_site{disk_num}_chamber{j}_minus30",
                    pos=pos_minus30,
                    rgba=[1, 1, 1, 1],
                    group=1,
                )

        for i in range(4):
            tendon_plus30 = self.mjcf_model.tendon.add(
                "spatial",
                name=f"{self.name}_tendon{i}_plus30",
                width=0.03,
                rgba=[.6, .6, .6, 1])

            tendon_minus30 = self.mjcf_model.tendon.add(
                "spatial",
                name=f"{self.name}_tendon{i}_minus30",
                width=0.03,
                rgba=[.6, .6, .6, 1])

            for j in range(self.num_disks):
                tendon_plus30.add(
                    "site", site=f"{self.name}_site{j}_chamber{i}_plus30")
                tendon_minus30.add(
                    "site", site=f"{self.name}_site{j}_chamber{i}_minus30")

    def _addBellow(self, bellows_num):
        tendon = self.mjcf_model.tendon.add(
            "spatial",
            name=f"{self.name}_tendon{bellows_num}",
            width=0.03,
            rgba=[.6, .6, .6, 0])

        for i in range(self.num_disks):
            tendon.add("site",
                       site=f"{self.name}_site{i}_chamber{bellows_num}")


if __name__ == "__main__":

    large_bellows = EightBellows("j0", 5, .2, .125, 1.0)
    medium_bellows = FourBellows("j1", 5, .2, .1, 1.0)
    small_bellows = FourBellows("j2", 5, .2, .08, 1.0)

    # I really don't like how the attach does things...
    large2medium = large_bellows.get_connection_site()
    large2medium.attach(medium_bellows.mjcf_model)
    medium2small = medium_bellows.get_connection_site()
    medium2small.attach(small_bellows.mjcf_model)

    # four_bellows = FourBellows("j1", 5, .2, .08, 1.0, parent_body=connection)

    # print(four_bellows)

    xml = large_bellows.mjcf_model.to_xml_string()

    #save xml
    with open("four_bellows.xml", "w") as f:
        f.write(xml)

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
