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
    #todo: change to set_gravity(model, "off" or "on")
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


    This function returns the contact forces AS FELT by body_name. See https://github.com/google-deepmind/mujoco/issues/679

    Since mujoco contact normal points from geom1 to geom2, the default force calculated by mujoco is the force FELT by geom2.
    If it so happens that geom1 is attached to "body_name", then this function returns the vector in the opposite direction.

    This function EXCLUDES contact with the ground.

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
        contact_geom_ids = [data.contact[i].geom[0], data.contact[i].geom[1]]
        # I don't care about contact with the ground
        if model.geom("world").id not in contact_geom_ids:
            # if any geoms involved in contact are attached to body we're interested in
            if not set(contact_geom_ids).isdisjoint(
                    set(geomid_attached_to_body)):

                body_geomid_in_contact = set(contact_geom_ids).intersection(
                    set(geomid_attached_to_body)).pop()

                F_CC_C = np.zeros(
                    6)  # wrench at contact point C expressed in contact frame
                f_C_W = np.zeros(
                    3)  # force at contact point C expressed in world frame
                mujoco.mj_contactForce(
                    model, data, i,
                    F_CC_C)  # doesn't throw error if i is out of range of ncon
                R_CW = data.contact[i].frame

                mujoco.mju_mulMatTVec3(f_C_W, R_CW, F_CC_C[:3])
                #if the geom attached to body_name is geom1, then the force is in the opposite direction since mujoco reports normal from geom 1 to geom 2
                if body_geomid_in_contact == data.contact[i].geom[0]:
                    f_C_W = -f_C_W

                # sometimes there are contacts that are active but don't have any force
                # see https://mujoco.readthedocs.io/en/stable/programming/simulation.html#contacts
                if np.linalg.norm(f_C_W) > 0:
                    contact_forces.append(f_C_W)

    if len(contact_forces) == 0:
        return np.zeros((0, 3))

    con = np.asarray(contact_forces).copy()

    return con


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
    return object_pose[:3].copy()


def get_box_vel(model, data):
    #todo: see mj_objectVelocity
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
    return object_vel[:3].copy()


def get_box_angvel(model, data):
    """
    Returns the angular velocity of the box in the world frame.

    Parameters:
    -----------
    model: mujoco_py.MjModel
        The MuJoCo model object.
    data: mujoco_py.MjData
        The MuJoCo data object containing the current simulation state.

    Returns:
    --------
    numpy.ndarray
        A 3D numpy array representing the angular velocity of the box in the world frame.
    """
    qvel_adr = model.joint(model.body("box").jntadr).dofadr.item()
    free_body_len = 6  # linear vel + angular vel
    object_vel = data.qvel[qvel_adr:qvel_adr + free_body_len]
    return object_vel[3:].copy()


def get_box_quat(model, data, scalar_first=True, positive_w=True):
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

    positive_w is set to True by default, since mujoco does not do anything about the quaternion double coverage issue.
    
    """
    qpos_adr = model.joint(model.body("box").jntadr).qposadr.item()
    free_body_len = 7  # position + quaternion
    object_pose = data.qpos[qpos_adr:qpos_adr + free_body_len]
    quat = object_pose[-4:].copy()  #[qw, qx, qy, qz]

    if positive_w:
        if quat[0] < 0:
            quat = -quat

    if scalar_first:
        return quat
    else:
        return np.roll(quat, -1)  # [qx, qy, qz, qw]


def get_elevator_cmd(model, data):
    return data.ctrl[model.actuator("elevator").id].copy()


def get_elevator_activation(model, data):
    '''
    returns [filtered vel cmd, filtered pos cmd]

    since the input is an overall position, and ruckig plugin filters it to provide these values. 
    '''
    actuator_id = model.actuator("elevator").id
    start_index = model.actuator(actuator_id).actadr.item()
    num_elements = model.actuator(actuator_id).actnum.item()

    actuator_values = data.act[start_index:start_index + num_elements]
    return actuator_values.copy()


def set_elevator_cmd(model, data, value):
    try:
        data.ctrl[model.actuator("elevator").id] = value
        return True
    except:
        return False


def get_elevator_height(model, data):
    '''returns joint coordinate of elevator in meters'''
    return data.qpos[model.joint("chest::linear_actuator").qposadr].copy()


def get_elevator_vel(model, data):
    return data.qvel[model.joint("chest::linear_actuator").dofadr].copy()


def get_joint_pressures(model, data, side, jointnum):
    act_adr = [
        model.actuator(f"{side}_arm::j{jointnum}::p{i}").actadr
        for i in range(4)
    ]

    pressures = np.asarray([data.act[adr][0] for adr in act_adr])

    return pressures.copy()


def set_joint_pressure_commands(model, data, side, jointnum,
                                pressure_commands):

    pressure_commands = np.clip(pressure_commands, 0, 400)

    actuator_ids = [
        model.actuator(f"{side}_arm::j{jointnum}::p{i}").id for i in range(4)
    ]
    for i, actuator_id in enumerate(actuator_ids):
        data.ctrl[actuator_id] = pressure_commands[i]


def get_joint_pressure_commands(model, data, side, jointnum):
    cmds = []
    for i in range(4):
        cmds.append(
            data.ctrl[model.actuator(f"{side}_arm::j{jointnum}::p{i}").id])
    return np.asarray(cmds).copy()


def get_tactile_image(model, data, side: Literal['left', 'right', 'chest'],
                      linknum: Literal[0, 1]):
    "note that contact between tactile sleeves and ground has been disabled in the xml file"
    if linknum == 0:
        tactile_img = np.zeros((64, 16))
        for i in range(64):
            for j in range(16):
                sensor_name = f"{side}_arm::link{linknum}::touch_{i}_{j}"
                taxel_force = data.sensordata[model.sensor(sensor_name).id]
                # fill column
                tactile_img[i, j] = taxel_force
    elif linknum == 1:
        tactile_img = np.zeros((64, 16))
        for i in range(64):
            for j in range(16):
                sensor_name = f"{side}_arm::link{linknum}::touch_{i}_{j}"
                taxel_force = data.sensordata[model.sensor(sensor_name).id]
                # fill column
                tactile_img[i, j] = taxel_force

    elif side == "chest":
        tactile_img = np.zeros((32, 30))
        for i in range(32):  # cols
            for j in range(30):  # rows
                # logic to deal with slanted sides
                if j <= (11 / 10) * i + 19 and j <= (-11 / 10) * i + 53:
                    sensor_name = f"chest::touch_{i}_{j}"
                    taxel_force = data.sensordata[model.sensor(sensor_name).id]
                    tactile_img[i, j] = taxel_force

    return tactile_img


def get_joint_angles(model, data, side, jointnum):
    joint_angle_sensor = data.sensor(f"{side}_arm::j{jointnum}").data
    return joint_angle_sensor[:2].copy()


def get_joint_vel(model, data, side, jointnum):
    joint_vel_sensor = data.sensor(f"{side}_arm::j{jointnum}").data
    return joint_vel_sensor[2:].copy()


def detect_box_touch(model, data):
    """
    returns true is box is in contact with anything besides the ground
    """
    if data.ncon > 0:
        for i in range(data.ncon):
            if model.geom("box").id in data.contact.geom:
                # box is being touched. Need to check if it is the ground or not
                if not model.geom("world").id in data.contact[i].geom:
                    return True
    return False


def detect_box_on_ground(model, data):
    box_id = model.geom("box").id
    world_id = model.geom("world").id

    for i in range(data.ncon):
        contact_geom = data.contact[i].geom
        if box_id in contact_geom and world_id in contact_geom:
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
        data.joint(f"{side}_arm::j{jointnum}::Jx_{i}").qpos = x_disk_angle
        data.joint(f"{side}_arm::j{jointnum}::Jy_{i}").qpos = y_disk_angle

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
            f"{side}_arm::j{jointnum}::Jx_{i}").dofadr] = x_disk_vel
        data.qvel[model.joint(
            f"{side}_arm::j{jointnum}::Jy_{i}").dofadr] = y_disk_vel

    mujoco.mj_forward(model, data)


def get_chest_position(model: mujoco.MjModel,
                       data: mujoco.MjData) -> ArrayLike:
    """
    This function returns the position of the chest in the world frame:
    [x,y,z] meters. 

    Returns:
        ArrayLike[float, float, float]: Position of the chest in the world frame [x,y,z] meters.
    """
    return data.geom("chest").xpos.copy()


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
    return data.geom(f"{side}_arm::link{linknum}").xpos.copy()


def get_disk_position(model, data, side: Literal['left', 'right'],
                      joint_num: Literal[0, 1, 2], disk_num: int):
    """
    This function returns the position and orientation of the specified disk in the world frame:

    Args:
        model (MjModel): mujoco model
        data (MjData): mujoco data
        side (Literal[left, right]): left or right arm
        joint_num (Literal[0, 1, 2]): Joint number [0, 1, 2]
        disknum (int): Disk number [0, num_disks). Specify -1 to get the last (meaning most distal) disk.

    Returns:
        Tuple[ArrayLike[float, float, float], ArrayLike[float, float, float, float]]: Position and orientation of the disk in the world frame.
    """

    num_disks_in_model = int(model.numeric("num_disks").data.item())
    if disk_num == -1:
        disk_num = num_disks_in_model - 1
    try:
        return data.geom(
            f"{side}_arm::j{joint_num}::disk{disk_num}").xpos.copy()
    except KeyError as e:
        error = f"Disk {disk_num} not found for {side} arm, joint {joint_num}. There are {num_disks_in_model} disks in the model and joint_num must be 0, 1, or 2."
        raise KeyError(error) from None


def get_disk_quat(model: mujoco.MjModel,
                  data: mujoco.MjData,
                  side: Literal['left', 'right'],
                  joint_num: Literal[0, 1, 2],
                  disk_num: int,
                  scalar_first=True) -> ArrayLike:
    """This function returns the orientation of the specified disk in the world frame.

    Args:
        model (MjModel): mujoco model
        data (MjData): mujoco data
        side (Literal[left, right]): left or right arm
        joint_num (Literal[0, 1, 2]): Joint number [0, 1, 2]
        disk_num (int): Disk number [0, num_disks). Specify -1 to get the last (meaning most distal) disk.
        scalar_first (bool, optional): If True, returns the quaternion in scalar-first format [qw, qx, qy, qz]. If False, returns the quaternion in vector-first format [qx, qy, qz, qw]. Defaults to True.

    Raises:
        KeyError: If the disk is not found for the specified side, joint_num, or disk_num.

    Returns:
        ArrayLike: Orientation of the disk in the world frame.
    """

    num_disks_in_model = int(model.numeric("num_disks").data.item())
    if disk_num == -1:
        disk_num = num_disks_in_model - 1
    try:
        xmat = data.geom(f"{side}_arm::j{joint_num}::disk{disk_num}").xmat
        xquat = np.zeros(4)
        mujoco.mju_mat2Quat(xquat, xmat)

        if scalar_first:
            return xquat
        else:
            return np.roll(xquat, -1)
    except KeyError as e:
        error = f"Disk {disk_num} not found for {side} arm, joint {joint_num}. There are {num_disks_in_model} disks in the model and joint_num must be 0, 1, or 2."
        raise KeyError(error) from None


def apply_wrench_to_body(model, data, body_name, force, torque):
    #point_on_body needs to be in world frame
    body_com = data.body(body_name).xipos

    #apply cartesian force and torque to body CoM, and transform to generalized forces qfrc_applied
    mujoco.mj_applyFT(model, data, force, torque, body_com,
                      model.body(body_name).id, data.qfrc_applied)


def clear_wrenches(model, data):
    data.qfrc_applied[:] = 0


def add_visual_geom(user_scn, geom_type, size, pos, mat, rgba):
    """
    Adds a visual geom to the user scene. 

    Args:
        user_scn (mujoco.mjvScene): User scene object
        geom_type (mujoco.mjtGeom): Type of geom to add
        size (ArrayLike): Size of the geom
        pos (ArrayLike): Position of the geom
        mat (ArrayLike): Rotation matrix of the geom
        rgba (ArrayLike): Color of the geom

    Returns:
        mujoco.mjvGeom: The geom object that was added to the user scene.
    """
    custom_geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        custom_geom,
        type=geom_type,
        size=size,
        pos=pos,
        mat=mat,
        rgba=rgba,
    )
    user_scn.ngeom += 1
    return custom_geom


def set_box_position(model, data, x=None, y=None, z=None):
    """
    Sets the position of the box in the world frame.

    Args:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.
        x,y,z (ArrayLike): Position of the box in the world frame.
    """

    box_pose = data.joint(model.body("box").jntadr).qpos

    if x is not None:
        box_pose[0] = x
    if y is not None:
        box_pose[1] = y
    if z is not None:
        box_pose[2] = z

    mujoco.mj_forward(model, data)


def set_box_quat(model,
                 data,
                 qw=None,
                 qx=None,
                 qy=None,
                 qz=None,
                 positive_w=True):
    """
    Sets the orientation of the box in the world frame.

    This function ensures that the quaternion w is positive, just for consistency.
    Mujoco does not do anything about the quaternion double coverage issue.

    Args:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.
        qw,qx,qy,qz (ArrayLike): Orientation of the box in the world frame.
    """
    box_pose = data.joint(model.body("box").jntadr).qpos

    quat = np.array([qw, qx, qy, qz])
    if positive_w:
        if quat[0] < 0:
            quat = -quat

    if qw is not None:
        box_pose[3] = quat[0]
    if qx is not None:
        box_pose[4] = quat[1]
    if qy is not None:
        box_pose[5] = quat[2]
    if qz is not None:
        box_pose[6] = quat[3]

    mujoco.mj_forward(model, data)


def _calc_cuboid_inertia_diag(mass, xsize, ysize, zsize):
    """
    Calculates the inertia tensor of a cuboid.

    Args:
        mass (float): Mass of the cuboid.
        xsize (float): Length of the cuboid in x.
        ysize (float): Width of the cuboid in y.
        zsize (float): Height of the cuboid in z.

    Returns:
        ArrayLike: Inertia tensor of the cuboid.
    """
    Ixx = (mass / 12) * (ysize**2 + zsize**2)
    Iyy = (mass / 12) * (xsize**2 + zsize**2)
    Izz = (mass / 12) * (xsize**2 + ysize**2)
    return np.array([Ixx, Iyy, Izz])


def set_box_size(mjSpec: mujoco.MjSpec, xsize=None, ysize=None, zsize=None):
    """
    Sets the size of the box. This function modifies the size of the box geom in the MuJoCo model, 
    hence it is necessary to pass the MuJoCo spec.

    Also recalculates the inertia tensor of the box.

    Args:
        mjSpec (mujoco.MjSpec): MuJoCo spec object
        xsize (float): Length of the box in x.
        ysize (float): Width of the box in y.
        zsize (float): Height of the box in z.
    """
    box = mjSpec.find_body("box")
    box_size = box.first_geom().size

    if xsize is not None:
        box_size[0] = xsize / 2
    if ysize is not None:
        box_size[1] = ysize / 2
    if zsize is not None:
        box_size[2] = zsize / 2

    #update inertia tensor with new geometry and same mass
    box.inertia = _calc_cuboid_inertia_diag(box.mass, *box_size)


def set_box_mass(mjSpec: mujoco.MjSpec, mass: float):
    """
    Sets the mass of the box. This function modifies the mass of the box body in the MuJoCo model, 
    hence it is necessary to pass the MuJoCo spec.

    Also recalculates the inertia tensor of the box.

    Args:
        mjSpec (mujoco.MjSpec): MuJoCo spec object
        mass (float): Mass of the box.
    """
    box = mjSpec.find_body("box")
    box.mass = mass

    #update inertia tensor with same geometry and new mass
    box_size = box.first_geom().size
    box.inertia = _calc_cuboid_inertia_diag(mass, *box_size)


def get_all_contact_wrenches(model, data):
    wrenches = []
    for i in range(data.ncon):
        F = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, F)

        wrenches.append(F)

    return np.asarray(wrenches).copy()


def check_arms_touching_ground(model, data):
    """
    Returns True if any part of the arms are touching the ground. 

    The box can touch the ground though, so this function returns false if only the box is touching the ground.
    """

    touching_ground = False
    for i in range(data.ncon):
        if model.geom("world").id in data.contact[i].geom:
            #something is touching the ground. If its NOT the box, then something else is toching the ground.
            if model.geom("box").id not in data.contact[i].geom:
                touching_ground = True
                break

    return touching_ground


def check_arm_base_collision(model, data):
    """
    Return true if any part of any arm is touching anything on the base. 

    The base body has multiple geoms attached to it, but all have "base::" prefixes in their names. 
    """

    for i in range(data.ncon):
        #get both geoms in contact
        geom1_id = data.contact[i].geom[0]
        geom2_id = data.contact[i].geom[1]
        geom1_name = model.geom(geom1_id).name
        geom2_name = model.geom(geom2_id).name

        # check if either geom is attached to the base body
        if "base::" in geom1_name or "base::" in geom2_name:
            # check if either geom is attached to the arms
            if "left_arm::" in geom1_name or "left_arm::" in geom2_name or \
                    "right_arm::" in geom1_name or "right_arm::" in geom2_name:
                return True

    return False


def check_arm_arm_collision(model, data):
    """
    Returns true if any part of any arm is touching anything on the other arm.

    Each arm has many bodies and geoms attached to it, but all have "left_arm::" or "right_arm::" prefixes in their names. 

    """

    for i in range(data.ncon):
        #get both geoms in contact
        geom1_id = data.contact[i].geom[0]
        geom2_id = data.contact[i].geom[1]
        geom1_name = model.geom(geom1_id).name
        geom2_name = model.geom(geom2_id).name

        #see which geom is attached to which arm
        if "left_arm::" in geom1_name and "right_arm::" in geom2_name:
            return True
        elif "right_arm::" in geom1_name and "left_arm::" in geom2_name:
            return True

    return False
