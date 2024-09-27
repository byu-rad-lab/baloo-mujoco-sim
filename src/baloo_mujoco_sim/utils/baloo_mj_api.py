import numpy as np
from dataclasses import dataclass
import mujoco
from typing import Literal
from numpy.typing import ArrayLike


@dataclass
class Jangles:
    j0: float
    j1: float
    j2: float
    j3: float
    j4: float
    j5: float

    def as_array(self):
        return np.array([self.j0, self.j1, self.j2, self.j3, self.j4, self.j5])


def disable_gravity(model):
    model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY.value


def set_mocap_pose(model, data, mocap_name, pos=None, quat=None):
    mocapid = model.body(mocap_name).mocapid

    if pos is not None:
        data.mocap_pos[mocapid, :] = pos

    if quat is not None:
        data.mocap_quat[mocapid, :] = quat


def set_mocap_size(model, data, mocap_name, size):
    # number of size varies depending on type. check type of geom and assert size is correct

    mocap_geom_type = model.geom(mocap_name).type
    if mocap_geom_type == mujoco.mjtGeom.mjGEOM_SPHERE.value:
        assert len(size) == 1, "Sphere size must be length 1"
    elif mocap_geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE.value:
        assert len(size) == 1 or len(
            size) == 2, "Capsule size must be length 1 or 2"
    elif mocap_geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value:
        assert len(size) == 3, "Ellipsoid size must be length 3"
    elif mocap_geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER.value:
        assert len(size) == 1 or len(
            size) == 2, "Cylinder size must be length 1 or 2"
    elif mocap_geom_type == mujoco.mjtGeom.mjGEOM_BOX.value:
        assert len(size) == 3, "Box size must be length 3"

    model.geom(mocap_name).size = size


def get_contact_forces_on_body(model, data, body_name):
    """
    Returns the contact forces acting on the specified body.

    Parameters:
    -----------
    model: mujoco_py.MjModel
        The MuJoCo model object.
    data: mujoco_py.MjData
        The MuJoCo data object containing the current simulation state.
    body_name: str
        The name of the body to get the contact forces for.

    Returns:
    --------
    numpy.ndarray
        A 2D numpy array representing the contact forces acting on the specified body in the world frame (ncon x 3).
    """
    geom_start_adr = model.body(body_name).geomadr.item()
    num_geoms_attached_to_body = model.body(body_name).geomnum.item()
    geomid_attached_to_body = np.linspace(
        geom_start_adr,
        geom_start_adr + num_geoms_attached_to_body,
        num_geoms_attached_to_body,
        dtype=int,
    )

    contact_forces = []
    for i in range(data.ncon):
        contact_geom_ids = [data.contact[i].geom1, data.contact[i].geom2]
        # I don't care about contact with the ground
        if model.geom("world").id not in contact_geom_ids:
            # if any geoms involved in contact are attached to body we're interested in
            if contact_geom_ids in geomid_attached_to_body:
                F_CC_C = np.zeros(
                    6)  # wrench at contact point C expressed in contact frame
                f_C_W = np.zeros(
                    3)  # force at contact point C expressed in world frame
                mujoco.mj_contactForce(
                    model, data, i,
                    F_CC_C)  # doesn't throw error if i is out of range of ncon
                R_CW = data.contact[i].frame
                mujoco.mju_rotVecMatT(f_C_W, F_CC_C[:3], R_CW)
                contact_forces.append(f_C_W)
    return np.asarray(contact_forces)


def get_box_position(model, data):
    """
    Returns the position of the center of mass of the box in the world frame.

    Parameters:
    -----------
    model: mujoco_py.MjModel
        The MuJoCo model object.
    data: mujoco_py.MjData
        The MuJoCo data object containing the current simulation state.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array representing the position of the center of mass of the box in the world frame.

    Note:
    -----
    Since mujoco requires floating bodies to be children of the world body, the joint space coords
    of the box are the same as the world coords. The pose data is stored in qpos as [x, y, z, qw, qx, qy, qz].
    """
    qpos_adr = model.joint(model.body("box").jntadr).qposadr.item()
    free_body_len = 7  # position + quaternion
    object_pose = data.qpos[qpos_adr:qpos_adr + free_body_len]
    return object_pose[:3]


def get_box_vel(model, data):
    """
    Returns the linear velocity of the box in the world frame.

    Parameters:
    -----------
    model: mujoco_py.MjModel
        The MuJoCo model object.
    data: mujoco_py.MjData
        The MuJoCo data object containing the current simulation state.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array representing the linear velocity of the box in the world frame.
    """
    qvel_adr = model.joint(model.body("box").jntadr).dofadr.item()
    free_body_len = 6  # linear vel + angular vel
    object_vel = data.qvel[qvel_adr:qvel_adr + free_body_len]
    return object_vel[:3]


def get_box_quat(model, data, scalar_first=True):
    """
    Returns the orientation of the frame attached to the center of mass of the box in the global frame.

    Parameters:
    -----------
    model: mujoco_py.MjModel
        The MuJoCo model object.
    data: mujoco_py.MjData
        The MuJoCo data object containing the current simulation state.

    Returns:
    --------
    numpy.ndarray
        A 4D numpy array (unit quaternion) representing the orientation of the frame attached to the center of mass of the box relative to the global frame.
        Note that the quaternion is in the order [w, x, y, z].

    Note:
    -----
    This returns the quaternion as specified here https://mujoco.readthedocs.io/en/latest/programming/simulation.html#coordinate-frames-and-transformations:~:text=To%20represent%203D,quaternions%20in%20MJCF.
    """
    qpos_adr = model.joint(model.body("box").jntadr).qposadr.item()
    free_body_len = 7  # position + quaternion
    object_pose = data.qpos[qpos_adr:qpos_adr + free_body_len]
    if scalar_first:
        return object_pose[-4:]  #[qw, qx, qy, qz]
    else:
        return np.roll(object_pose[-4:], -1)  # now [qx, qy, qz, qw]


def get_elevator_cmd(model, data):
    return data.ctrl[model.actuator("elevator").id]


def set_elevator_cmd(model, data, value):
    try:
        data.ctrl[model.actuator("elevator").id] = value
        return True
    except:
        return False


def get_elevator_height(model, data):
    return data.qpos[model.joint("linear_actuator").qposadr]


def get_elevator_vel(model, data):
    return data.qvel[model.joint("linear_actuator").dofadr]


def get_joint_pressures(model, data, side, jointnum):
    pressures = []
    for i in range(4):
        pressures.append(
            data.act[model.actuator(f"{side}_j{jointnum}_p{i}").actadr])
    return np.asarray(pressures).squeeze()


def set_joint_pressure_commands(model, data, side, jointnum,
                                pressure_commands):

    pressure_commands = np.clip(pressure_commands, 0, 400)
    for i in range(4):
        data.ctrl[model.actuator(
            f"{side}_j{jointnum}_p{i}").id] = pressure_commands[i]


def get_joint_pressure_commands(model, data, side, jointnum):
    cmds = []
    for i in range(4):
        cmds.append(data.ctrl[model.actuator(f"{side}_j{jointnum}_p{i}").id])
    return np.asarray(cmds)


def get_tactile_image(model, data, side: Literal['left', 'right', 'chest'],
                      linknum: Literal[0, 1]):
    "note that contact between tactile sleeves and ground has been disabled in the xml file"
    if linknum == 0:
        tactile_img = np.zeros((64, 16))
        for i in range(64):
            for j in range(16):
                sensor_name = f"{side}_link{linknum}_{i}_{j}_touch"
                taxel_force = data.sensordata[model.sensor(sensor_name).id]
                # fill column
                tactile_img[i, j] = taxel_force
    elif linknum == 1:
        tactile_img = np.zeros((64, 16))
        for i in range(64):
            for j in range(16):
                sensor_name = f"{side}_link{linknum}_{i}_{j}_touch"
                taxel_force = data.sensordata[model.sensor(sensor_name).id]
                # fill column
                tactile_img[i, j] = taxel_force

    elif side == "chest":
        tactile_img = np.zeros((32, 30))
        for i in range(32):  # cols
            for j in range(30):  # rows
                # logic to deal with slanted sides
                if j <= (11 / 10) * i + 19 and j <= (-11 / 10) * i + 53:
                    sensor_name = f"chest_{i}_{j}_touch"
                    taxel_force = data.sensordata[model.sensor(sensor_name).id]
                    tactile_img[i, j] = taxel_force

    return tactile_img


def get_joint_angles(model, data, side, jointnum):
    joint_angle_sensor = data.sensor(f"{side}_j{jointnum}").data
    return joint_angle_sensor[:2]


def get_joint_vel(model, data, side, jointnum):
    joint_vel_sensor = data.sensor(f"{side}_j{jointnum}").data
    return joint_vel_sensor[2:]


def detect_box_touch(model, data):
    """
    returns true is box is in contact with anything besides the ground
    """
    if data.ncon > 0:
        for i in range(data.ncon):
            if (data.contact[i].geom1 == model.geom("box").id
                    or data.contact[i].geom2 == model.geom("box").id):
                # box is being touched. Need to check if it is the ground or not
                if not (data.contact[i].geom1 == model.geom("world").id
                        or data.contact[i].geom2 == model.geom("world").id):
                    return True
    return False


def set_joint_angles(model, data, side, jointnum, jangles):
    """
    Evenly divides jangles between the disks of the joint and sets the joint angles in data.qpos.

    Also calls mj_forward to put all of mjData in a consistent state. For more info
    see https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#main-simulation 
    Args:
        model (MjModel): mujoco model
        data (MjData): mujoco data
        side (string): left or right
        jointnum (int): 0, 1, or 2
        jangles (np array): np.array([qx,qy])
    """
    num_disks = int(model.numeric("num_disks").data.item())

    x_disk_angle = jangles[0] / (num_disks - 1)
    y_disk_angle = jangles[1] / (num_disks - 1)

    #find jointid for each disk joint angles in data.qpos that corresponds to the jointnum
    for i in range(num_disks - 1):
        data.joint(f"{side}_j{jointnum}_Jx_{i}").qpos = x_disk_angle
        data.joint(f"{side}_j{jointnum}_Jy_{i}").qpos = y_disk_angle

    mujoco.mj_forward(model, data)


def set_joint_velocities(model, data, side, jointnum, jvel):
    """
    Evenly divides jvel between the disks of the joint and sets the joint velocities in data.qvel.

    Also calls mj_forward to put all of mjData in a consistent state. For more info
    see https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#main-simulation 
    Args:
        model (MjModel): mujoco model
        data (MjData): mujoco data
        side (string): left or right
        jointnum (int): 0, 1, or 2
        jvel (np array): np.array([qxd,qyd])
    """
    num_disks = int(model.numeric("num_disks").data.item())

    x_disk_vel = jvel[0] / (num_disks - 1)
    y_disk_vel = jvel[1] / (num_disks - 1)

    #find jointid for each disk joint angles in data.qpos that corresponds to the jointnum
    for i in range(num_disks - 1):
        data.qvel[model.joint(
            f"{side}_j{jointnum}_Jx_{i}").dofadr] = x_disk_vel
        data.qvel[model.joint(
            f"{side}_j{jointnum}_Jy_{i}").dofadr] = y_disk_vel

    mujoco.mj_forward(model, data)


def get_chest_position(model: mujoco.MjModel,
                       data: mujoco.MjData) -> ArrayLike:
    """
    This function returns the position of the chest in the world frame:
    [x,y,z] meters. 

    Returns:
        ArrayLike[float, float, float]: Position of the chest in the world frame [x,y,z] meters.
    """
    return data.geom("chest").xpos


def get_chest_velocity(model, data):
    raise NotImplementedError


def get_link_position(model, data, side: Literal['left', 'right'],
                      linknum: Literal[0, 1]) -> ArrayLike:
    """
    This function returns the position of the specified link in the world frame:
    [x,y,z] meters.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        side (Literal[left, right]): left or right arm
        linknum (Literal[0, 1]): Link 0 or 1. 

    Returns:
        ArrayLike[float, float, float]: [x,y,z] meters in world frame.
    """
    return data.geom(f"{side}_link{linknum}").xpos
